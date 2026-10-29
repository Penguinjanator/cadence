"""Bounded recall storage preserves episode semantics and outstanding array snapshots."""

import numpy as np
import pytest

import cadence as cd


def two_slots():
    brain = cd.Brain.compose(
        2, 2, modules=(4,), episodic=False, recall=True,
        recall_temperature=0.25, recall_recency=0.0,
    )
    brain.act([[1.0, 0.0]], greedy=True)
    brain.act([[0.0, 1.0]], greedy=True)
    return brain


def attention_reference(recall, address):
    # Scalar dot products, independent of the memory's attention/cache implementation.
    logits = np.array([[sum(a * k for a, k in zip(query, key, strict=True))
                        / recall.temperature for key in slots]
                       for query, slots in zip(address, recall.keys, strict=True)])
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def test_retuned_temperature_and_saved_continuation_read_the_current_weights(tmp_path):
    brain = two_slots()
    address = brain.recall.keys[:, 0].copy()
    before = brain.recall.weights(address).copy()
    brain.retune(recall_temperature=1.0)
    expected = attention_reference(brain.recall, address)
    assert not np.array_equal(before, expected)
    np.testing.assert_array_equal(brain.recall.weights(address), expected)
    restored = cd.Brain.load(brain.save(tmp_path / "retuned.npz"))
    np.testing.assert_array_equal(restored.recall.weights(address), expected)


def test_current_key_contents_are_read_after_an_in_place_expert_edit():
    brain = two_slots()
    recall = brain.recall
    address = recall.keys[:, 0].copy()
    before = recall.weights(address).copy()
    recall.keys[:, 0] *= -1.0
    expected = attention_reference(recall, address)
    assert not np.array_equal(before, expected)
    np.testing.assert_array_equal(recall.weights(address), expected)


@pytest.mark.parametrize("capacity", [1, 2, 7])
def test_successors_and_eviction_match_independent_stream_histories(capacity):
    recall = cd.Brain.compose(
        4, 2, modules=(8,), episodic=False, recall=True, recall_capacity=capacity,
    ).recall
    recall._empty(3)
    rng = np.random.default_rng(825)
    histories = [[] for _ in range(3)]
    for tick in range(25):
        # Not every stream holds every slot; episode endings can also differ per stream.
        closed = rng.random(3) < 0.3
        recall.ended = closed.copy()
        before = {name: getattr(recall, name) for name in ("keys", "values", "nexts", "held")}
        saved = {name: array.copy() for name, array in before.items()}
        keys, values = rng.normal(size=(2, 3, 4))
        if tick % 4 == 1:
            holds = rng.random(3) < 0.5
            recall._append(keys, holds, values)
        else:
            holds = np.ones(3, dtype=bool)
            recall._observe(keys, values)
            for row, history in enumerate(histories):
                if not closed[row]:
                    for slot in reversed(history):
                        if slot[3]:
                            slot[2] = values[row].copy()
                            break
        for row, history in enumerate(histories):
            history.append([keys[row], values[row], np.zeros(4), holds[row]])
            del history[:-capacity]
        for name, member in (("keys", 0), ("values", 1), ("nexts", 2), ("held", 3)):
            expected = np.array([[slot[member] for slot in history] for history in histories])
            np.testing.assert_array_equal(getattr(recall, name), expected, err_msg=name)
            np.testing.assert_array_equal(before[name], saved[name], err_msg="snapshot " + name)
