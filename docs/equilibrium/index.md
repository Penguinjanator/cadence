# The population solver

`cadence.experimental.equilibrium` builds one connected graph of bounded
patches that repair disagreement together. Each patch holds a state, ports,
retained relations and a local prediction. You choose population sizes and the
ports they read, teach through the same joint repair, then measure free
behaviour. Outputs are selected patch states from a qualified solve. Import it
with `from cadence.experimental.equilibrium import Cortex`.

A population reads live states of other populations with `inputs=`; an
observer population (`Cortex.observer`) also reads their exact current errors
with `observes=`. Returning influence enters the same energy and the same
qualification check as every other edge.

This solver is separate from the default brain,
[`Brain.compose`](../brain.md) (see [one continuing equilibrium brain](../world-model.md)).
It has its own energy and stationarity equations, explicit `History` in place
of the working trace and associative stores, and no planner.

## A short learning path

1. [Quickstart](QUICKSTART.md): build a brain, teach a relation, test fresh
   predictions without targets, then save and resume.
2. [Brain design](BRAIN_DESIGN.md): choose observations, temporal context,
   wiring, connected capacity and a cost budget; teach and diagnose a behaviour.
3. [Live operation](LIVE.md): drive an actual body, learn from actual outcomes
   and check retained skills.
4. [Execution and cost](ACCELERATION.md): devices, precision, batch repair,
   independent lives and query and learning work.

The [specification](SPECIFICATION.md) states the equations, qualification and
refusal rules; the [API reference](REFERENCE.md) lists every class and argument.

## Scope and limits

- The package is experimental. APIs and saved formats may change with no
  cross-version compatibility or migration layer. Save the source and the
  complete application continuation (body state, pending outcomes) for any
  experiment you need to reproduce; checkpoints validate the implementation
  sources they were written with.
- Observer error readback (`observes=`) is implemented and tested. Its
  advantage on a task, automatic recruitment on surprise and separate observer
  clocks are open research; ordinary connected populations are the control.
- Current disagreement, the miss of an earlier forecast and an unmet goal are
  separate quantities. A qualified stationary state can predict reality poorly,
  so check useful behaviour separately. Learned relations persist, and
  competing learning can overwrite them.
- `Reinforcement` uses `credit_horizon=1` by default; a larger horizon changes
  return construction and replay cost and supplies no learned planning.

## Pages and examples

| Read | Purpose |
| --- | --- |
| [Quickstart](QUICKSTART.md) | Build, teach, test, save and resume; the isolated-population refusal |
| [Brain design](BRAIN_DESIGN.md) | Wiring, teaching rows, capacity and diagnosis |
| [Live operation](LIVE.md) | Actual bodies, reward learning, outcome ownership and retention |
| [Execution and cost](ACCELERATION.md) | Devices, batching, parallel lives and cost |
| [Specification](SPECIFICATION.md) | Equations, qualification, refusal and continuation |
| [API reference](REFERENCE.md) | Classes, parameters, methods and diagnostics |
| [Agent guide](AGENTS.md) | Application workflow for coding agents |
| [Folder entry](README.md) | Short entry for the package |
| [Runnable examples](../../examples/equilibrium/README.md) | Learning, body control, history, reward, batching and cost |

Open work is tracked in the [issues](https://github.com/muellerberndt/cadence/issues).
