# Agent guide: the population solver

## Scope and reading order

This subtree documents `cadence.experimental.equilibrium`: one connected graph
of populations whose patches settle together under a joint energy, with
explicit `History` as its temporal memory. It shares no memory or learning
code with `Brain.compose`. The [root agent guide](../../AGENTS.md) holds the
repository-wide rules. Link to a topic's page instead of restating it.

| Topic | Page |
| --- | --- |
| What the solver is, scope and limits | [index](index.md#scope-and-limits) |
| First brain, operations table, example commands | [quickstart](QUICKSTART.md) |
| Body boundary, wiring, teaching, diagnosis | [brain design](BRAIN_DESIGN.md) |
| Live use, reward learning, ownership, saving | [live operation](LIVE.md) |
| Devices, batching, cost | [execution and cost](ACCELERATION.md) |
| Equations and qualification | [specification](SPECIFICATION.md) |
| Signatures, defaults, refusals | [API reference](REFERENCE.md) |

## Application rules

- The application owns sensor meaning, action decoding and outcome units.
  Future targets and privileged environment state are never actor inputs
  ([body boundary](BRAIN_DESIGN.md#define-the-bodys-information-and-outcomes)).
- Start from two coupled populations and add capacity when a measurement asks
  for it. `build()` refuses a population that settles with no other population
  and a layout whose populations or patches split into disconnected groups
  ([connectivity and cost](BRAIN_DESIGN.md#connectivity-and-cost)).
- Teach through `observe`, `observe_batch` or `bootstrap`. Measured labels are
  witnesses (`source="witness"`, the default); derived teaching values pass
  `source="estimate"`. Clamps on `settle` and `step` condition that call only
  and never teach ([clamps](REFERENCE.md#values-clamps-and-events)).
- Test with the teacher disconnected and outputs free. Check `qualified` before
  acting and `accepted` before counting learning. A stationary state can carry
  disagreement and is no evidence of task success
  ([qualification](SPECIFICATION.md#repair-and-qualification)).
- Record each forecast before its outcome and keep the action actually executed.
  Discrete reward learning goes through `Reinforcement`
  ([ownership](LIVE.md#preserve-execution-and-feedback-ownership)).
- Save the brain with preprocessing, body state, history and RNG; for reward
  learning save the `Reinforcement` snapshot so the pending action survives.
  Serialize calls to one brain; `LiveController` keeps a caller responsive
  ([saving](LIVE.md#save-a-whole-life-and-run-the-small-gates),
  [responsiveness](LIVE.md#keep-the-body-responsive)).
- Observer readback (`observes=`) is experimental, and a population's name or
  depth establishes no capability
  ([recursive observers](BRAIN_DESIGN.md#experimental-recursive-observer-settlement)).

## Documentation and implementation checks

Keep examples on the public API. Prefer a link to one runnable example over a
new wrapper or an application-specific core abstraction. Every Python fence on
these pages runs, all fences of a page in one namespace, unless a
`<!-- not-run: reason -->` marker directly precedes it. Put signatures in
`text` fences.

`tests/equilibrium/test_documentation.py` checks [REFERENCE.md](REFERENCE.md)
against the code: every exported name appears, and parameter order, kinds and
defaults match for `Cortex`, `bootstrap`, `History`, `LearningProgress`,
`Reinforcement`, `LiveController`, `slew` and every public method of those five
classes and `Brain`. A signature change updates the reference in the same
change. After a documentation change, run from the repository root:

```sh
PYTHONPATH=src python -m pytest -q tests/equilibrium/test_documentation.py tests/equilibrium/test_packaging.py
```

Runtime changes need all of `tests/equilibrium` and the root guide's
[review and verification](../../AGENTS.md#review-and-verification) checks.
