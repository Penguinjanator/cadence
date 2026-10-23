"""Isolated readers for retained campaigns; live prerequisite guards stay strict.

Only freshly loaded checker modules receive these dependency views. No public
class, runtime module, receipt, model parameter or saved policy is changed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from cadence.receipts import Receipt


class _ConfigView:
    def __init__(self, actual):
        self.actual = actual

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def to_dict(self):
        value = self.actual.to_dict()
        if type(value.get("initial_damping")) is not int or value["initial_damping"] != 0:
            raise ValueError("historical checker requires the neutral numerical default")
        return {key: item for key, item in value.items() if key != "initial_damping"}


class _FounderView:
    """Read-only expected-record view, used only inside one offline arithmetic check."""
    def __init__(self, actual, record):
        self.actual, self.record = actual, record
        self.learner = actual.learner
        if "initial_damping" not in record["learning_config"]:
            self.learner = _LearnerView(actual.learner)

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def describe(self):
        value = self.actual.describe()
        genes = value["genes"]
        if "learning_initial_damping" not in self.record["resolved"]["genes"]:
            if (type(genes.get("learning_initial_damping")) is not int
                    or genes["learning_initial_damping"] != 0):
                raise ValueError("historical checker requires the neutral numerical default")
            del genes["learning_initial_damping"]
        return value


class _LearnerView:
    def __init__(self, actual):
        self.actual, self.config = actual, _ConfigView(actual.config)

    def __getattr__(self, name):
        return getattr(self.actual, name)


class Session:
    """A private module tree, with exact historical source bytes for each nested receipt."""
    def __init__(self, root, resolve, reference):
        self.root, self.resolve = Path(root), resolve
        self.reference = Receipt.read(reference)
        valid, message = Receipt.verify(reference, sources=resolve(self.reference))
        if not valid:
            raise ValueError("historical reference custody differs: " + message)
        self.stack, self.installed = [], set()
        self.receipts = SimpleNamespace(
            read=Receipt.read, build=Receipt.build, verify=self.verify_receipt)

    @property
    def current(self):
        return self.stack[-1] if self.stack else self.reference

    def bound_paths(self, paths):
        recorded = dict(self.resolve(self.current))
        return [(name, recorded.get(name, path)) for name, path in paths]

    def dependency_problem(self, path, receipt):
        if receipt.digest == self.current.digest:
            return None
        try:
            relative = Path(path).resolve().relative_to(self.root).as_posix()
        except ValueError:
            return "historical dependency is outside the retained repository"
        parent = {row["path"]: row["sha256"] for row in self.current.source["files"]}
        if (relative not in parent
                or parent[relative] != hashlib.sha256(Path(path).read_bytes()).hexdigest()):
            return "historical dependency bytes differ from the parent receipt"
        return None

    def verify_receipt(self, path, *, sources=None, check=None):
        receipt = Receipt.read(path)
        if sources is not None and [name for name, _ in sources] != [
                row["path"] for row in receipt.source["files"]]:
            return False, "historical source inventory differs"
        problem = self.dependency_problem(path, receipt)
        if problem:
            return False, problem
        try:
            bound = self.resolve(receipt)
        except (ValueError, OSError) as error:
            return False, str(error)
        self.stack.append(receipt)
        try:
            return Receipt.verify(path, sources=bound, check=check)
        finally:
            self.stack.pop()

    def load(self, filename):
        spec = importlib.util.spec_from_file_location(
            "offline_attention_" + Path(filename).stem,
            self.root / "benchmarks/attention" / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if filename in ("minimal_binding_campaign.py", "single_relevance_campaign.py",
                        "neutral_variation_campaign.py"):
            capacity = {"minimal_binding_campaign.py": "minimal-binding-capacity.json",
                        "single_relevance_campaign.py": "single-relevance-capacity.json",
                        "neutral_variation_campaign.py":
                            "neutral-variation-capacity.json"}[filename]
            module.configure(self.root, self.root / "benchmarks/attention/results" / capacity)
        expected = ({module.KIND} if filename == "heads_continuation.py" else {
            stage["receipt_kind"] for stage in module.declared()["stages"].values()})
        if self.reference.kind not in expected:
            raise ValueError("historical reference kind differs from its checker")
        self.install(module)
        return module

    def install(self, module):
        if id(module) in self.installed:
            return
        self.installed.add(id(module))
        if hasattr(module, "Receipt"):
            module.Receipt = self.receipts
        for name in ("relevance", "admission"):
            child = getattr(module, name, None)
            if child is not None and hasattr(child, "__file__"):
                self.install(child)
        if hasattr(module, "context"):
            context = module.context()
            for name in ("owner", "base", "campaign", "admission", "relevance"):
                child = getattr(context, name, None)
                if child is not None:
                    self.install(child)
        if hasattr(module, "h1_verifier"):
            original_h1 = module.h1_verifier

            def h1_verifier():
                child = original_h1()
                self.install(child)
                return child

            module.h1_verifier = h1_verifier
        if hasattr(module, "sources"):
            original_sources = module.sources
            module.sources = lambda *args, **kwargs: self.bound_paths(
                original_sources(*args, **kwargs))
        if hasattr(module, "numerical_check"):
            original_numerical = module.numerical_check
            module.numerical_check = lambda result, protocol: original_numerical(
                result, protocol, historical=True)
        if Path(module.__file__).name == "single_relevance_campaign.py":
            original_check, factory = module.check, module.make_brain
            original_panel_check = module.panel_check

            def panel_check(panel, causal=False, *, brain=None, arm=None):
                # Native boundary checks depend on the actual oracle class.
                actual = brain.actual if isinstance(brain, _FounderView) else brain
                return original_panel_check(panel, causal, brain=actual, arm=arm)

            module.panel_check = panel_check

            def check(body):
                enclosing_factory = module.make_brain
                records = {(row["arm"], row["seed"]): row for row in body["runs"]
                           if row.get("status") != "unrun_budget"}
                for row in records.values():
                    for values, key in ((row["learning_config"], "initial_damping"),
                                        (row["resolved"]["genes"], "learning_initial_damping")):
                        if key in values and (type(values[key]) is not int or values[key] != 0):
                            return "historical numerical policy is not the neutral default"
                module.make_brain = lambda arm, seed: _FounderView(
                    factory(arm, seed), records[(arm, seed)])
                try:
                    return original_check(body)
                finally:
                    module.make_brain = enclosing_factory

            module.check = check
        if hasattr(module, "verify"):
            original_verify = module.verify

            def verify(path):
                if path is None:
                    return original_verify(path)
                receipt = Receipt.read(path)
                problem = self.dependency_problem(path, receipt)
                if problem:
                    return False, problem
                self.stack.append(receipt)
                try:
                    return original_verify(path)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    return False, str(error)
                finally:
                    self.stack.pop()

            module.verify = verify
