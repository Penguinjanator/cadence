# Core tutorials

These small programs explain the library. The six interactive task brains live in
[cadence-examples](https://github.com/muellerberndt/cadence-examples).
Install Cadence, then run any tutorial from this checkout with
`python examples/<name>.py`. Only NumPy is required.

| Program | Mechanism and useful check |
|---|---|
| [generic_brain.py](generic_brain.py) | Continuous `GenericBrain.step`: demonstration, action and observed reward in one loop. Compares repeated and salient experience after clearing transient memory. |
| [coupled_brain.py](coupled_brain.py) | `assemble` connects sensory and motor regions; `equilibrate` checks the joint equations within a bounded work budget. |
| [deliberation.py](deliberation.py) | `imagine` compares isolated futures using a learned terminal evaluator. A wrong-world-model control fails despite successful search. Only real outcomes teach. |
| [brain_patterns.py](brain_patterns.py) | `ActivityMonitor` reads uncertainty and requests deeper bounded, pruned adversarial imagination. |
| [certified_memory.py](certified_memory.py) | `certificate` reads the contraction rate and bounds the remaining settling error; `PatternSeparator` keeps correlated keys from interfering in a delta-rule record. |
| [worked_update.py](worked_update.py) | Inspect one local contrastive synaptic update and its teaching signal. |
| [two_blobs.py](two_blobs.py) | Minimal supervised example with held-out evaluation. |
| [half_center.py](half_center.py) | Recurrent oscillation: some useful dynamics keep moving instead of reaching a static fixed point. |
| [ring_protocol.py](ring_protocol.py) | Follow the local message and state updates in a recurrent circuit. |

Separate phase calls in the mechanism tutorials expose the numerical rule.
Applications can use the continuous loop while neuronal settling and synaptic
plasticity operate on different timescales. A read alone is not a new observation.
The [continuous-learning guide](../docs/continuous.md) describes feedback timing,
consolidation, resets and checkpoints. The [brain-design guide](../docs/design.md)
explains how to choose components and capacity for a task.
