from __future__ import annotations

import numpy as np
import pytest

from cadence.content_memory import ContentMemory


def test_selects_by_noisy_content_without_labels_or_slot_ids() -> None:
    memory = ContentMemory(3, 2, 3)
    memory.observe(np.eye(3)[:2], np.eye(2))
    query = np.array([[0.99, 0.1, 0], [0.1, 0.99, 0]])
    keys, values, ages = memory.keys.copy(), memory.values.copy(), memory.last_write.copy()
    assert np.array_equal(memory.recall(query), np.eye(2))
    assert np.array_equal(memory.keys, keys)
    assert np.array_equal(memory.values, values)
    assert np.array_equal(memory.last_write, ages)
    assert memory.writes == 2
    assert memory.select(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))[0].tolist() == [-1, -1]


def test_value_labels_cannot_select_address_and_prototypes_actually_learn() -> None:
    a, b = (ContentMemory(3, 2, 4, match=0.5, key_rate=0.5) for _ in range(2))
    x = np.array([[1.0, 0.0, 0.0], [1.0, 0.4, 0.0], [0.0, 0.0, 1.0]])
    a.observe(x, np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    b.observe(x, np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]))
    assert np.array_equal(a.keys, b.keys)
    assert a.size == b.size == 2
    assert a.keys[0, 1] > 0  # neither a fixed random key nor an unchanged exemplar
    assert np.array_equal(a.select(x)[0], b.select(x)[0])


def test_revisions_and_capacity_replace_only_one_record() -> None:
    memory = ContentMemory(3, 2, 2)
    memory.observe(np.eye(3)[:2], np.eye(2))
    memory.observe(np.array([[1.0, 0.0, 0.0]]), np.array([[0.2, 0.8]]))
    assert np.array_equal(memory.recall(np.eye(3)[:2]), [[0.2, 0.8], [0.0, 1.0]])
    memory.observe(np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 1.0]]))
    assert memory.size == 2 and memory.evictions == 1
    assert memory.select(np.eye(3))[0].tolist() == [0, -1, 1]
    assert memory.to_dict()["mutable_bytes"] == 2 * (3 + 2 + 1) * 8


def test_aliasing_remains_an_explicit_failure() -> None:
    memory = ContentMemory(2, 2, 10)
    x = np.array([[1.0, 0.0], [1.0, 0.0]])
    memory.observe(x, np.eye(2))
    assert memory.size == 1
    assert np.array_equal(memory.recall(x), [[0.0, 1.0], [0.0, 1.0]])


def test_zero_mask_and_bad_input_do_not_mutate_memory() -> None:
    memory = ContentMemory(2, 1, 2)
    memory.observe(np.eye(2), np.ones((2, 1)), np.zeros(2, dtype=bool))
    memory.observe(np.zeros((1, 2)), np.ones((1, 1)))
    assert memory.size == memory.writes == 0
    with pytest.raises(ValueError):
        memory.observe(np.eye(2), np.array([[1.0], [np.nan]]))
    with pytest.raises(ValueError):
        memory.observe(np.eye(2), np.ones((2, 1)), np.ones(2))
    assert memory.size == memory.writes == 0
    memory.observe(np.array([[1e300, 1e300]]), np.array([[1.0]]))
    assert np.allclose(memory.recall(np.array([[1e-300, 1e-300]])), 1)


@pytest.mark.parametrize(
    "kwargs", [{"capacity": 0}, {"inputs": True}, {"match": np.nan}, {"value_rate": 1.1}]
)
def test_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ContentMemory(**({"inputs": 2, "outputs": 1, "capacity": 2} | kwargs))
