"""Acquisition controls for the named-fact selection instrument: can the same graph, taught
the same way, acquire AND, OR, XOR and the selection task itself when both facts are exposed?

A declared development comparison beside ``selective_recall.py``: same encoding, trial
structure, teaching path and work accounting; the target function of the two exposed facts
is the only change. Selection on the unequal-fact cases is XOR of the query bit and fact A,
so a graph that cannot acquire XOR cannot acquire selection. Width, depth and rounds vary as
declared comparisons; nothing here is a runtime change or a protocol of the instrument.

    PYTHONPATH=src python benchmarks/relevance/acquisition_controls.py --out DIR --workers 3
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import tempfile
import time
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import selective_recall as sr  # noqa: E402

TASKS = ("and", "or", "xor", "mux")
BRAINS = {"m32": [32], "m64": [64], "m32x16": [32, 16], "m64x32": [64, 32]}
ROUNDS = (128, 512, 2048)
FOUNDERS = (0, 1, 2, 3, 4, 5)
NEUTRAL_EVENTS = 1


def targets(task: str, rows: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """The answer the task requires for facts (A, B) and the query that names one of them."""
    a, b = rows[:, 0], rows[:, 1]
    if task == "mux":
        return rows[np.arange(len(rows)), queries].copy()
    if task == "and":
        return a & b
    if task == "or":
        return a | b
    if task == "xor":
        return a ^ b
    raise ValueError(task)


def train(
    brain: Any, task: str, order: np.ndarray, work: sr.Work, teaching: str = "every",
    exposed: bool = True,
) -> dict[str, Any]:
    """The instrument's training loop with the task's targets; both facts exposed at the query
    (``exposed``), or absent, the instrument's trace arm.

    ``teaching`` ``every`` is the instrument's: one lesson on every case of the round, then the
    free action. ``surprise`` acts first and teaches only the rows whose answer was wrong (the
    recall chamber's surprise rule), then acts again; a round without a wrong row has no lesson.
    """
    rounds = 0
    stop = None
    lessons = 0
    for permutation in order:
        rows = sr.CASES[permutation]
        prefix, query, _ = sr.inputs(rows[:, :3], rows[:, 3], NEUTRAL_EVENTS, exposed=exposed)
        target = targets(task, rows, rows[:, 3])
        brain.reset()
        for observation in prefix:
            if work.act(brain, observation) is None:
                stop = "refused_prefix"
                break
        if stop is not None:
            break
        if teaching == "surprise":
            answer = work.act(brain, query)
            if answer is None:
                stop = "refused_query"
                break
            wrong = np.flatnonzero(np.asarray(answer).reshape(-1) != target)
            if len(wrong):
                work.teach(brain, query[wrong], target[wrong])
                lessons += 1
        else:
            work.teach(brain, query, target)
            lessons += 1
        if work.act(brain, query) is None:
            stop = "refused_query"
            break
        rounds += 1
    return {"completed": rounds == len(order), "rounds": rounds, "stop": stop, "lessons": lessons}


def evaluate(brain: Any, task: str, work: sr.Work, exposed: bool = True) -> dict[str, Any]:
    """The instrument's forked evaluation: every history to the pre-query state, both queries."""
    answers = np.full((8, 2), -1, dtype=np.int64)
    expected = np.stack(
        [targets(task, sr.FACTS, np.full(8, q, dtype=np.int64)) for q in range(2)], axis=1
    )
    with tempfile.TemporaryDirectory(prefix="cadence-relevance-controls-") as directory:
        directory = Path(directory)
        from cadence import Brain  # noqa: PLC0415

        prefix_brain = Brain.load(brain.save(directory / "initial.npz"))
        prefix_brain.reset()
        prefix, _, _ = sr.inputs(sr.FACTS, np.zeros(8, dtype=np.int64), NEUTRAL_EVENTS)
        complete = True
        for observation in prefix:
            if work.act(prefix_brain, observation) is None:
                complete = False
                break
        state = prefix_brain.save(directory / "prequery.npz")
        if complete:
            for q in range(2):
                branch = Brain.load(state)
                _, query, _ = sr.inputs(
                    sr.FACTS, np.full(8, q, dtype=np.int64), NEUTRAL_EVENTS, exposed=exposed
                )
                answer = work.act(branch, query)
                if answer is None:
                    complete = False
                else:
                    answers[:, q] = answer
    valid = answers >= 0
    correct = (answers == expected) & valid
    unequal = sr.FACTS[:, 0] != sr.FACTS[:, 1]
    return {
        "completed": complete,
        "answers": answers.tolist(),
        "expected": expected.tolist(),
        "accuracy": float(correct.mean()),
        "unequal_fact_accuracy": float(correct[unequal].mean()),
        "per_query_accuracy": correct.mean(axis=0).tolist(),
        "unanswered": int((~valid).sum()),
    }


def job(
    spec: tuple[str, str, int, int, str, float | None, bool, dict[str, float]],
) -> dict[str, Any]:
    task, brain_name, rounds, seed, teaching, eta, exposed, trace = spec
    from cadence import Brain  # noqa: PLC0415

    began = time.monotonic()
    rng = np.random.default_rng(np.random.SeedSequence([31014, seed]))
    order = np.stack([rng.permutation(len(sr.CASES)) for _ in range(rounds)])
    options: dict[str, Any] = {}
    if eta is not None:
        # the teaching contrast's step and its bias step at the founder's ratio of one tenth
        options["learning_eta"], options["learning_eta_bias"] = eta, eta / 10
    options.update({f"working_memory_{name}": value for name, value in trace.items()})
    brain = Brain.compose(
        sr.INPUTS, 2, seed=seed, modules=BRAINS[brain_name], episodic=False, **options
    )
    work = sr.Work()
    trained = train(brain, task, order, work, teaching, exposed)
    evaluation = evaluate(brain, task, work, exposed)
    return {
        "task": task, "brain": brain_name, "modules": BRAINS[brain_name], "rounds": rounds,
        "seed": seed, "neutral_events": NEUTRAL_EVENTS, "teaching": teaching, "exposed": exposed,
        "learning_eta": eta, "trace_genes": trace, "resolved_working_memory": {
            k: v for k, v in brain.describe()["working_memory"].items()
            if k in ("decay", "amplitude", "scale")
        }, "resolved_learning": {
            k: v for k, v in brain.describe()["learning"].items() if k in ("eta", "eta_bias")
        },
        "training": trained, "evaluation": evaluation, "work": asdict(work),
        "elapsed_seconds": round(time.monotonic() - began, 3),
    }


def digests() -> dict[str, str]:
    out = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sr.sources()}
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out["benchmarks/relevance/acquisition_controls.py"] = me
    return out


def table(rows: list[dict[str, Any]]) -> str:
    lines = ["| task | brain | rounds | accuracy, mean [per founder] | unequal-fact accuracy "
             "| founders at or above 0.95 |", "| --- | --- | --- | --- | --- | --- |"]
    keys = sorted({(r["task"], r["brain"], r["rounds"]) for r in rows},
                  key=lambda k: (TASKS.index(k[0]), list(BRAINS).index(k[1]), k[2]))
    for task, brain_name, rounds in keys:
        cell = sorted(
            (r for r in rows if (r["task"], r["brain"], r["rounds"]) == (task, brain_name, rounds)),
            key=lambda r: r["seed"],
        )
        acc = [r["evaluation"]["accuracy"] for r in cell]
        une = [r["evaluation"]["unequal_fact_accuracy"] for r in cell]
        lines.append(
            f"| {task} | {brain_name} | {rounds} | {np.mean(acc):.3f} "
            f"[{' '.join(f'{a:.2f}' for a in acc)}] | {np.mean(une):.3f} "
            f"[{' '.join(f'{u:.2f}' for u in une)}] | {sum(a >= 0.95 for a in acc)}/{len(acc)} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS), choices=TASKS)
    parser.add_argument("--brains", nargs="*", default=list(BRAINS), choices=list(BRAINS))
    parser.add_argument("--rounds", nargs="*", type=int, default=list(ROUNDS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--teaching", choices=("every", "surprise"), default="every")
    parser.add_argument(
        "--learning-eta", type=float, default=None,
        help="the teaching contrast's step; the composed founder when omitted",
    )
    parser.add_argument("--query", choices=("exposed", "trace"), default="exposed",
                        help="both facts exposed at the query, or absent (the trace arm)")
    for gene in ("decay", "amplitude", "scale"):
        parser.add_argument(f"--trace-{gene}", type=float, default=None,
                            help=f"working_memory_{gene}; the composed founder when omitted")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    trace = {g: v for g, v in (("decay", args.trace_decay), ("amplitude", args.trace_amplitude),
                               ("scale", args.trace_scale)) if v is not None}
    specs = [(t, b, r, s, args.teaching, args.learning_eta, args.query == "exposed", trace)
             for t, b, r, s in itertools.product(args.tasks, args.brains, args.rounds, args.seeds)]
    done = {p.stem for p in args.out.glob("*.json") if p.stem != "summary"}
    todo = [s for s in specs if f"{s[0]}-{s[1]}-{s[2]}-{s[3]}" not in done]
    print(f"{len(todo)} runs to go, {len(done)} done", flush=True)
    import cadence  # noqa: PLC0415

    with get_context("spawn").Pool(max(1, min(args.workers, len(todo) or 1))) as pool:
        for row in pool.imap_unordered(job, todo):
            name = f"{row['task']}-{row['brain']}-{row['rounds']}-{row['seed']}"
            (args.out / f"{name}.json").write_text(json.dumps(row, indent=1))
            e = row["evaluation"]
            print(
                f"  {name}: accuracy {e['accuracy']:.3f} unequal {e['unequal_fact_accuracy']:.3f} "
                f"answers {e['answers']} ({row['elapsed_seconds']}s)",
                flush=True,
            )
    rows = [json.loads(p.read_text()) for p in args.out.glob("*.json") if p.stem != "summary"]
    summary = {"cadence_version": cadence.__version__, "numpy_version": np.__version__,
               "sources": digests(), "runs": len(rows), "teaching": args.teaching,
               "learning_eta": args.learning_eta, "query": args.query, "trace_genes": trace,
               "table": table(rows)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(summary["table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
