# Experimental fixed connectivity

`cadence.experimental.PartitionedTemporalPatchNet` applies fixed boolean
connection masks to one [`TemporalPatchNet`](temporal.md); its coordinate groups
are parts of one jointly repaired temporal path. The masks are supplied
architecture for controlled experiments: the model learns no modules, roles or
specialization; compare it against a same-width dense control
([cost and controls](#cost-and-controls)).

## What the masks mean

The causal recurrence is the ordinary one:

```text
h[t] = A @ tanh(h[t-1]) + B @ input[t]
y[t] = C @ tanh(h[t])
```

`PartitionedTemporalPatchNet(inputs, hidden, outputs, *, masks=None, **options)`
takes `masks` as a dict of boolean arrays `A` `(hidden, hidden)`, `B`
`(hidden, inputs)` and `C` `(outputs, hidden)`; `options` go to
`TemporalPatchNet`. Any other key set, dtype or shape raises `ValueError`. A
true `A[row, column]` connects the previous state's `column` to the current
state's `row`, so recurrent communication spans one path step (the model's time
coordinate, never a solver iteration), with no instantaneous coupling or separate
clock. `masks=None` means all-true masks, which reproduce the dense model's
phases, updates and carried state exactly.

Initialization draws the dense parameters and zeros forbidden entries without
renormalizing, so `initial_radius` describes the dense draw. Gradients are
masked before any candidate update or backtracking, and `set_parameters` raises
`ValueError` on a nonzero forbidden entry. The masks are immutable; the `masks`
property returns copies and `trainable_parameter_count` counts permitted entries.

## A small executable example

`two_group_masks(context_hidden, motor_hidden, inputs, outputs, *,
context_inputs, motor_inputs, cross_coupling=True)` builds two coordinate
groups with the same dynamics; the names describe supplied roles. Within-group
recurrence is dense, `cross_coupling=True` permits both delayed cross-group
directions, and only the motor group reaches the outputs. Input index sets may
overlap or be empty; out-of-range or duplicate indices raise `ValueError`. For
another topology, pass your own three arrays.

```python
import numpy as np

from cadence.experimental import PartitionedTemporalPatchNet, two_group_masks

masks = two_group_masks(
    context_hidden=3, motor_hidden=3, inputs=2, outputs=1,
    context_inputs=[0], motor_inputs=[1], cross_coupling=True,
)
net = PartitionedTemporalPatchNet(2, 6, 1, masks=masks, seed=7)
assert net.trainable_parameter_count == 36 + 6 + 3
inputs = np.zeros((1, 4, 2))
inputs[0, 0, 0] = 1.0  # one observed context pulse
inputs[0, :, 1] = [0.0, 0.1, -0.1, 0.2]
target = np.array([[[0.0], [0.1], [0.2], [0.1]]])

result = net.observe(inputs, target, beta=0.01, rate=0.1, backtrack=True)
assert result.free.converged
for key, values in net.parameters().items():
    assert np.all(values[~net.masks[key]] == 0)

restored = PartitionedTemporalPatchNet.restore(net.snapshot())
expected = net.imagine(inputs).output
np.testing.assert_array_equal(restored.imagine(inputs).output, expected)
```

`observe`, `advance`, `imagine`, `plan`, `reset` and `readback` keep the
[temporal contracts](temporal.md); `plan` adjusts the selected input ports under
fixed parameters and masks. A permitted connection does not guarantee
acquisition, so inspect the returned phase and learning outcomes.

## Checkpoints and compatibility

`snapshot()` records the masks under the format `cadence-fixed-connectivity/1`.
`PartitionedTemporalPatchNet.restore(snapshot)` and `PartitionedTemporalPatchNet.load(path)`
validate the masks and the zero forbidden entries and raise `ValueError` otherwise; a restore
through the base `TemporalPatchNet` drops mask enforcement.
[`TemporalMemory.observe`](temporal-memory.md) raises `TypeError` for this subclass before
changing either object; `TemporalMemory.protect` collects its constraints, but no protected
step uses them.

## Cost and controls

Masks reduce the trainable entries. The allocated arrays and the dense
hidden-width message solve keep their full size, so equal width costs what the
dense model costs ([cost and memory](temporal.md#cost-and-memory)). Zeroing
connections changes the initial recurrence spectrum, and input routing changes
what each group sees. Compare against a same-width dense control, counting
permitted parameters, mask bytes and phase work ([controls](evaluation.md)).

## Tests

`tests/test_partitioned.py` covers dense parity, finite-difference gradients,
legal backtracked updates, failed detuning, checkpoint continuation, private
planning and the `TemporalMemory` rejection. The hash-bound fixture in
`tests/fixtures/partitioned_phrase/` checks one real update against a frozen
reference.
