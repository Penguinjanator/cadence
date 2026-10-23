"""Actual immutable-bank interventions, without constructing or settling a brain."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadence import Brain

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "minimal_binding_bank_subject", HERE / "minimal_binding_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


@pytest.fixture
def context(monkeypatch):
    runtime = os.environ.get("CADENCE_ATTENTION_RUNTIME")
    if runtime is None:
        candidate = HERE.parents[1]
        if not (candidate / "benchmarks/attention/query_founder_campaign.py").is_file():
            pytest.skip("set CADENCE_ATTENTION_RUNTIME to the source-bound runtime")
        runtime = candidate

    def forbidden(*args, **kwargs):
        pytest.fail("immutable-bank test attempted a brain construction or neural phase")

    monkeypatch.setattr(Brain, "compose", forbidden)
    ctx = subject.configure(runtime)
    monkeypatch.setattr(ctx.admission.heads.Graph, "equilibrate", forbidden)
    return ctx


def boundary(ctx, copies):
    rows = ctx.task.panel("all")
    prefix, query = ctx.task.events(rows)
    values = np.tile(np.stack(prefix, axis=1), (1, 1, copies)) / np.sqrt(copies)
    heard = 0.75  # Nonunit heard makes independently correct K replacement observable.
    keys = heard * values
    held, ages = np.ones((8, 3), dtype=bool), np.asarray([2.0, 1.0, 0.0])
    width = copies * 19
    memory_type = ctx.admission.heads.Memory
    members = tuple(
        memory_type(
            query=np.arange(2 * head * width, (2 * head + 1) * width),
            read=np.arange((2 * head + 1) * width, (2 * head + 2) * width),
            keys=keys,
            values=values,
            held=held,
            ages=ages,
            amplitude=1.5,
            temperature=0.25,
            mode="agreement",
            empty_anchor=True,
        )
        for head in range(2)
    )
    bank = ctx.admission.heads.MultiMemory(members)
    brain = SimpleNamespace(port_copies=copies, recall=SimpleNamespace(heard=heard))
    drive = np.zeros((8, 4 * width))
    for member in bank.memories:
        drive[:, member.query] = np.tile(query, (1, copies)) / np.sqrt(copies)
    return rows, query, drive, bank, brain


@pytest.mark.parametrize("copies", [1, 2])
@pytest.mark.parametrize("name,condition", [(0, "flip_A"), (1, "flip_B")])
def test_donor_changes_only_actual_named_keys_and_values_across_both_orders(
    context,
    copies,
    name,
    condition,
):
    rows, query, drive, bank, brain = boundary(context, copies)
    originals = {field: getattr(bank, field).copy() for field in ("keys", "values", "held", "ages")}
    query_before, drive_before = query.copy(), drive.copy()
    donor = context.task.flipped(rows, name)
    prefix, donor_query = context.task.events(donor)
    expected_values = np.tile(np.stack(prefix, axis=1), (1, 1, copies)) / np.sqrt(copies)
    changed = subject.bank_change(bank, condition, brain, rows)

    np.testing.assert_array_equal(changed.values, expected_values)
    np.testing.assert_array_equal(changed.keys, brain.recall.heard * expected_values)
    np.testing.assert_array_equal(changed.held, originals["held"])
    np.testing.assert_array_equal(changed.ages, originals["ages"])
    np.testing.assert_array_equal(donor_query, query_before)
    np.testing.assert_array_equal(query, query_before)
    np.testing.assert_array_equal(drive, drive_before)
    for field, before in originals.items():
        np.testing.assert_array_equal(getattr(bank, field), before)

    selected = np.zeros((8, 3), dtype=bool)
    positions = rows[:, 2] if name == 0 else 1 - rows[:, 2]
    selected[np.arange(8), positions] = True
    assert np.all(np.any(changed.values != bank.values, axis=2) == selected)
    assert np.all(np.any(changed.keys != bank.keys, axis=2) == selected)
    np.testing.assert_array_equal(changed.values[~selected], bank.values[~selected])
    np.testing.assert_array_equal(changed.keys[~selected], bank.keys[~selected])
    for head, member in enumerate(changed.memories):
        for field in ("keys", "values", "held", "ages"):
            assert getattr(member, field) is getattr(changed.memories[0], field)
            assert not getattr(member, field).flags.writeable
        for field in ("query", "read"):
            np.testing.assert_array_equal(
                getattr(member, field), getattr(bank.memories[head], field)
            )
            assert not getattr(member, field).flags.writeable
        for field in ("amplitude", "temperature", "mode", "null", "recency", "empty_anchor"):
            assert getattr(member, field) == getattr(bank.memories[head], field)


@pytest.mark.parametrize("condition,name", [("flip_A", 0), ("flip_B", 1)])
@pytest.mark.parametrize("malformation", ["missing", "duplicate"])
def test_donor_refuses_a_missing_or_ambiguous_named_witness_before_mutating_bank(
    context,
    condition,
    name,
    malformation,
):
    rows, _, _, bank, brain = boundary(context, 1)
    first = bank.memories[0]
    keys, values, held, ages = (
        getattr(first, field).copy() for field in ("keys", "values", "held", "ages")
    )
    position = int(rows[0, 2]) if name == 0 else 1 - int(rows[0, 2])
    if malformation == "missing":
        held[0, position] = False
    else:
        keys = np.concatenate((keys, keys[:, position : position + 1]), axis=1)
        values = np.concatenate((values, values[:, position : position + 1]), axis=1)
        held = np.concatenate((held, np.ones((8, 1), dtype=bool)), axis=1)
        ages = np.concatenate((ages, [3.0]))
    invalid = replace(
        bank,
        memories=tuple(
            replace(
                member,
                keys=keys,
                values=values,
                held=held,
                ages=ages,
            )
            for member in bank.memories
        ),
    )
    before = {field: getattr(invalid, field).copy() for field in ("keys", "values", "held", "ages")}
    with pytest.raises(ValueError, match="exactly one witnessed named fact"):
        subject.bank_change(invalid, condition, brain, rows)
    for field, original in before.items():
        np.testing.assert_array_equal(getattr(invalid, field), original)


def test_declared_non_donor_bank_shams_preserve_joint_data_custody(context):
    rows, query, drive, bank, brain = boundary(context, 1)
    query_before, drive_before = query.copy(), drive.copy()
    for operation in ("neutral_sham", "reorder", "erase_0", "erase_1"):
        changed = subject.bank_change(bank, operation, brain, rows)
        if operation == "reorder":
            for field in ("keys", "values", "held"):
                np.testing.assert_array_equal(
                    getattr(changed, field), getattr(bank, field)[:, ::-1]
                )
            np.testing.assert_array_equal(changed.ages, bank.ages[::-1])
        elif operation == "neutral_sham":
            np.testing.assert_array_equal(changed.keys[:, :2], bank.keys[:, :2])
            np.testing.assert_array_equal(changed.values[:, :2], bank.values[:, :2])
            np.testing.assert_array_equal(changed.keys[:, 2], bank.keys[:, 2] / 2)
            np.testing.assert_array_equal(changed.values[:, 2], bank.values[:, 2] / 2)
            np.testing.assert_array_equal(changed.held, bank.held)
        else:
            name = int(operation[-1])
            mask = (bank.values[:, :, 0] > 0) & (bank.values[:, :, 5 + name] > 0)
            np.testing.assert_array_equal(changed.held, bank.held & ~mask)
            np.testing.assert_array_equal(changed.keys, bank.keys)
            np.testing.assert_array_equal(changed.values, bank.values)
        for field in ("keys", "values", "held", "ages"):
            assert getattr(changed.memories[0], field) is getattr(changed.memories[1], field)
            assert not getattr(changed.memories[0], field).flags.writeable
    np.testing.assert_array_equal(query, query_before)
    np.testing.assert_array_equal(drive, drive_before)
