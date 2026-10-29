"""Current evidence binds the executed model; historical evidence keeps its actual scope."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cadence as cd
from cadence.receipts import source_manifest

HERE = Path(__file__).resolve().parent
NAMES = {"held_heads": "held-heads.json", "look_back": "look-back.json",
         "own_action": "own-action.json", "copy_tokens": "copy-tokens.json",
         "contextual_choice": "contextual-choice-v0870.json"}


def module(name):
    spec = importlib.util.spec_from_file_location("custody_" + name, HERE / (name + ".py"))
    subject = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = subject
    spec.loader.exec_module(subject)
    return subject


def readings(subject, name):
    """Synthetic checker input derived from historical ratios, never a new measured result."""
    body = copy.deepcopy(cd.Receipt.read(HERE / "results" / NAMES[name]).body)
    body["configuration"] = subject.configuration()
    body["execution"] = subject.evidence["environment"]()
    for row in body.get("founders", body.get("runs", [])):
        row["source"] = source_manifest(subject.sources())
        row["execution"] = body["execution"]
        if name == "held_heads":
            row["protocol"] = {"lessons": 1024, "recall_temperature": 0.25}
    if name == "held_heads":
        body["schema"] = "cadence-held-heads/2"
        for founder in body["founders"]:
            for arm in founder["arms"].values():
                for split in ("trained", "held_out"):
                    row = arm[split]
                    row.update(trials=96, correct=round(row["accuracy"] * 96),
                               distinct_facts=row["distinct_top"],
                               role_distinct_top=row["distinct_top"])
    if name == "contextual_choice":
        for founder in body["founders"]:
            for panel in founder["panels"].values():
                panel.update(trials=32, correct=round(panel["accuracy"] * 32),
                             per_context_correct={key: round(value * 16) for key, value
                                                  in panel["per_context_accuracy"].items()})
    return body


@pytest.mark.parametrize("name", NAMES)
def test_new_receipts_bind_every_runtime_module_and_refuse_omissions(tmp_path, name):
    subject = module(name)
    files = subject.sources()
    names = {name for name, _ in files}
    expected = {path.relative_to(subject.ROOT).as_posix()
                for path in (subject.ROOT / "src/cadence").glob("*.py")}
    assert expected <= names
    assert {"pyproject.toml", "benchmarks/provenance.py"} <= names
    body = readings(subject, name)
    assert subject.check(body) is None
    path = cd.Receipt.build(body["schema"], body, files).write(tmp_path / "complete.json")
    assert cd.Receipt.verify(path, sources=files, check=subject.check)[0]
    omitted = [(key, path) for key, path in files if key != "src/cadence/fused.py"]
    path = cd.Receipt.build(body["schema"], body, omitted).write(tmp_path / "omitted.json")
    assert not cd.Receipt.verify(path, sources=files, check=subject.check)[0]


@pytest.mark.parametrize("name", NAMES)
def test_current_checkers_refuse_false_protocols_modes_and_census(name):
    subject = module(name)
    body = readings(subject, name)
    assert subject.check(body) is None
    for key, value in (("configuration", {"not_the_protocol": True, "lessons": 0}),
                       ("execution", {})):
        altered = copy.deepcopy(body)
        altered[key] = value
        assert subject.check(altered) is not None
    altered = copy.deepcopy(body)
    altered["execution"]["environment"]["CADENCE_FUSED"] = "0"
    altered["execution"]["effective_fused"] = True
    assert subject.check(altered) is not None
    key = "founders" if "founders" in body else "runs"
    altered = copy.deepcopy(body)
    altered[key].append(copy.deepcopy(altered[key][0]))
    assert subject.check(altered) is not None


@pytest.mark.parametrize("name", NAMES)
def test_historical_reader_does_not_admit_relabelled_or_altered_receipts(tmp_path, name):
    subject = module(name)
    path = HERE / "results" / NAMES[name]
    assert subject.verify_historical(path)[0]
    body = cd.Receipt.read(path).body
    body["configuration"] = {"lessons": 0}
    sources = subject.evidence["historical_sources"](subject.ROOT, cd.Receipt.read(path))
    altered = cd.Receipt.build(body["schema"], body, sources).write(tmp_path / "changed.json")
    assert not subject.verify_historical(altered)[0]


def test_whole_bank_head_metric_rejects_shared_neutral_peak():
    subject = module("held_heads")
    task, n = subject.Task("report"), 96
    first, second = np.zeros((n, 13)), np.zeros((n, 13))
    first[:, 0] = second[:, 0] = 0.99
    first[:, 1], first[:, 4] = 0.009, 0.001
    second[:, 1], second[:, 4] = 0.001, 0.009
    trial = SimpleNamespace(recall=SimpleNamespace(attended_heads=(first, second)))
    task.answer = lambda brain, bank, patterns: task.labels(patterns)
    result = subject.evaluate(task, trial, {}, held_out=False, n=n)
    assert result["role_distinct_top"] == 1.0
    assert result["distinct_top"] == result["distinct_facts"] == 0.0


def test_distinct_slots_of_the_same_fact_are_not_distinct_facts():
    subject = module("held_heads")
    task, n = subject.Task("report"), 96
    first, second = np.zeros((n, 13)), np.zeros((n, 13))
    first[:, 1], second[:, 2] = 1.0, 1.0
    trial = SimpleNamespace(recall=SimpleNamespace(attended_heads=(first, second)))
    task.answer = lambda brain, bank, patterns: task.labels(patterns)
    result = subject.evaluate(task, trial, {}, held_out=False, n=n)
    assert result["distinct_top"] == 1.0 and result["distinct_facts"] == 0.0
