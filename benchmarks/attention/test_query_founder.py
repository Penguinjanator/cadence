"""Constructor-only preservation of the ordinary query initializer; no solves."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from cadence.receipts import canonical_json

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("query_founder_tests", HERE / "query_founder.py")
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)
ARMS = ("h2", "h1", "h1_wide", "h1_wide_ports", "h1_more_steps")


def recipe():
    return json.loads((HERE / "heads-founder-extended-protocol.json").read_text())


def moments(brain):
    return {
        name: getattr(brain.learner, name).copy()
        for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
    }


def random_state(brain):
    return canonical_json(
        [brain.rng.bit_generator.state, brain.basal_ganglia.rng.bit_generator.state]
    )


def seam_mask(brain):
    pre, post = brain.connectome.pre, brain.connectome.post
    queries = np.concatenate(brain.head_queries)
    association = brain.association_index
    return (np.isin(pre, queries) & np.isin(post, association)) | (
        np.isin(post, queries) & np.isin(pre, association)
    )


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("seed", (0, 1))
def test_q0_reproduces_all_ten_frozen_initial_parameter_digests(arm, seed):
    results = json.loads((HERE / "results/heads-relevance-founder-extended.json").read_text())[
        "body"
    ]
    expected = next(r for r in results["runs"] if r["arm"] == arm and r["seed"] == seed)
    original = subject.admission.make_brain(arm, seed, recipe())
    graph, config, rng = original.brain, original.describe(), random_state(original)
    record = subject.initialize(original, seed=seed, query_scale=0)
    assert original.brain is graph
    assert (
        record["parameters_before"] == record["parameters_after"] == expected["initial_parameters"]
    )
    assert original.describe() == config and random_state(original) == rng
    assert not record["applied"] and record["outside_query_before"] == record["outside_query_after"]


@pytest.mark.parametrize("arm", ARMS)
def test_q1_changes_only_existing_reciprocal_query_efficacies(arm):
    brain = subject.admission.make_brain(arm, 0, recipe())
    graph, connectome = brain.brain, brain.connectome
    before = {name: getattr(graph, name).copy() for name in ("efficacy", "bias", "log_gain")}
    optimizer, rng, config = moments(brain), random_state(brain), brain.describe()
    query_ports = tuple(q.copy() for q in brain.head_queries)
    read_ports = tuple(q.copy() for q in brain.head_reads)
    bank = object()
    graph.memory = bank
    seam = seam_mask(brain)
    record = subject.initialize(brain, seed=0, query_scale=1)
    assert type(brain.brain) is type(graph) and brain.brain.memory is bank
    assert brain.connectome is connectome and brain.brain.connectome is connectome
    np.testing.assert_array_equal(brain.brain.efficacy[~seam], before["efficacy"][~seam])
    assert np.count_nonzero(brain.brain.efficacy[seam]) == int(seam.sum())
    for name in ("bias", "log_gain"):
        np.testing.assert_array_equal(getattr(brain.brain, name), before[name])
    for name, value in optimizer.items():
        np.testing.assert_array_equal(getattr(brain.learner, name), value)
    assert brain.learner.updates == brain.learner.contrast_updates == 0
    assert random_state(brain) == rng and brain.describe() == config
    for actual, expected in zip(brain.head_queries, query_ports, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(brain.head_reads, read_ports, strict=True):
        np.testing.assert_array_equal(actual, expected)
    lookup = {
        (int(a), int(b)): i
        for i, (a, b) in enumerate(zip(connectome.pre, connectome.post, strict=True))
    }
    for index in np.flatnonzero(seam):
        reverse = lookup[(int(connectome.post[index]), int(connectome.pre[index]))]
        assert brain.brain.efficacy[index] == brain.brain.efficacy[reverse]
    assert record["applied"] and record["outside_query_before"] == record["outside_query_after"]
    assert record["query_synapses"] == int(seam.sum())


def test_ordinary_draws_are_deterministic_independent_and_use_child_streams():
    a = subject.make_brain("h2", 0, recipe(), query_scale=1)
    b = subject.make_brain("h2", 0, recipe(), query_scale=1)
    np.testing.assert_array_equal(a.brain.efficacy, b.brain.efficacy)
    record = a.query_initialization
    assert record == b.query_initialization
    assert [h["seed_entropy"] for h in record["heads"]] == [[0, 0], [0, 1]]
    assert [h["seed_spawn_key"] for h in record["heads"]] == [[0], [0]]
    assert (
        record["heads"][0]["ordinary_projection_digest"]
        != record["heads"][1]["ordinary_projection_digest"]
    )
    assert record["parameters_before"] != record["parameters_after"]


@pytest.mark.parametrize("scale", (0, 1))
def test_plain_exposed_factory_is_unchanged_and_reports_inapplicability(scale):
    original = subject.admission.make_brain("exposed", 0, recipe())
    graph, config, rng = original.brain, original.describe(), random_state(original)
    record = subject.initialize(original, seed=0, query_scale=scale)
    assert (
        original.brain is graph and original.describe() == config and random_state(original) == rng
    )
    assert not record["applicable"] and not record["applied"] and not record["heads"]
    assert record["parameters_before"] == record["parameters_after"]


def test_supplied_oracle_preserves_its_class_and_uses_the_same_initializer():
    brain = subject.make_brain("supplied_reads", 0, recipe(), query_scale=1)
    assert isinstance(brain, subject.admission.FixedReadBrain)
    assert brain.query_initialization["applied"] and brain.heads == 2
    assert brain.recall.address == 0 and brain.empty_anchor


@pytest.mark.parametrize(
    "value", (True, False, np.bool_(True), -1, 0.5, float("nan"), float("inf"), "1", None)
)
def test_invalid_initializer_values_are_refused_before_factory(value):
    def forbidden(*args, **kwargs):
        raise AssertionError("factory must not run")

    with pytest.raises(ValueError, match="query_scale"):
        subject.make_brain("h2", 0, recipe(), query_scale=value, factory=forbidden)


def test_explicit_factory_dependency_and_single_fresh_initialization():
    calls = []

    def factory(*args):
        calls.append(args)
        return subject.admission.make_brain(*args)

    brain = subject.make_brain("h1", 0, recipe(), query_scale=1, factory=factory)
    assert len(calls) == 1
    before = subject.parameter_digest(brain.brain)
    with pytest.raises(ValueError, match="fresh"):
        subject.initialize(brain, seed=0, query_scale=1)
    assert subject.parameter_digest(brain.brain) == before


def test_nonfresh_or_contact_mismatch_refuses_before_mutation():
    brain = subject.admission.make_brain("h2", 0, recipe())
    before = subject.parameter_digest(brain.brain)
    brain.learner.updates = 1
    with pytest.raises(ValueError, match="fresh"):
        subject.initialize(brain, seed=0, query_scale=1)
    assert subject.parameter_digest(brain.brain) == before
    brain.learner.updates = 0
    brain.head_queries = (brain.head_queries[0][:-1], brain.head_queries[1])
    # A malformed port descriptor cannot silently initialize only part of a seam.
    with pytest.raises(ValueError):
        subject.initialize(brain, seed=0, query_scale=1)
    assert subject.parameter_digest(brain.brain) == before


def test_gene_defaults_and_parent_protocol_are_not_mutated():
    protocol = recipe()
    before = deepcopy(protocol)
    subject.make_brain("h2", 0, protocol, query_scale=1)
    assert protocol == before
