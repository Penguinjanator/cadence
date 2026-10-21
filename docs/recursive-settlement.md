# Recursive observers in one equilibrium

`PatchNet.recursive` builds a base population and a stack of observer populations in one
reciprocal `PatchNet` graph. Every settle solves `x = F(x; drive)` for the complete state
`x = (inputs, base, observer_1, ..., observer_D, outputs)` with one rate-neuron rule, so
each observer reads the evolving state, feeds back into it and is changed by that feedback
in the same solve. The result is an ordinary `PatchNet`: solving, teaching, checkpoints
and refusals follow the [PatchNet guide](patchnet.md).

## Choose the depth and widths

`PatchNet.recursive(inputs, layers, outputs, *, seed=0, coupling=1.0, config=None,
backend="cpu", device=None, **runtime_options)` takes positive integer port sizes and a
nonempty list of positive integer widths; `seed` is a nonnegative integer and `coupling`
finite and positive. `config=None` means `LearnerConfig(nudge="quadratic")`; any other
config must use `nudge="quadratic"`. `backend` and `device` select the
[backend](backends.md). `runtime_options` go to the `PatchNet` constructor (`steps=512`,
`chunk=32`, `tolerance=1e-5`, `source_capacity=256`, `solver="local"`,
`refinement_steps=64`, `context_strength`, `context_mask`; see the [PatchNet guide](patchnet.md)).

| Population | Members | Joined in both directions to |
| --- | --- | --- |
| `input`, `output` | the ports | `layer_0` and every observer |
| `layer_0` | the base, `layers[0]` neurons | inputs, outputs, every observer |
| `layer_k`, k ≥ 1 | observer level k, `layers[k]` neurons | inputs, outputs, `layer_0` and every other observer level |

Neuron indices run `input`, `layer_0`, `output`, `layer_1`, ... `observer` names all
levels from 1 up and `hidden` adds `layer_0`; these names serve per-level interventions
and measurements. There are no contacts inside a population and none from input to
output. `[12]` is the shallow control, `[12, 8, 4]` one base and two observer levels,
`[32] * 9` one base and eight. Inputs are soft drives whose values stay fixed during a
solve while the input neurons receive feedback. `read` returns the output neurons'
activity in the same equilibrium; there is no answer head outside the graph.

Symmetric random weights from `seed` are scaled by one factor so the largest absolute
incoming row sum equals `coupling`. With `tanh(v/2)` neurons (slope at most 0.5) the free
map has Lipschitz bound `coupling / 2`, and the damped local step (`dt = 0.5`) contracts
at rate `1 - dt * (1 - 0.5 * coupling)`: 0.75 at the default, 0.95 at `coupling=1.8`, no
certificate from 2 up. Learning and nudges change the equations, so read
`cd.certificate(net.brain)` on current parameters and check every phase
([settling certificates](certificate.md)).

With `I` inputs, `O` outputs and widths `L_0 ... L_D` there are
`2 * (I*L_0 + L_0*O + sum_{k>=1} L_k * (I + O + L_0 + ... + L_{k-1}))` directed contacts;
each reciprocal pair shares one weight and every neuron has a trainable bias (initially 0).
Storage and settle work grow with the contact count.
`recursive(1371, [24, 8], 6)` has 1,409 neurons, 88,512 contacts and 45,665 trainable
scalars.

```python
import numpy as np
import cadence as cd

net = cd.PatchNet.recursive(
    inputs=3, layers=[12, 8, 4], outputs=2, seed=7,
    coupling=1.0, steps=1024, tolerance=1e-9,
    config=cd.LearnerConfig(nudge="quadratic", beta=1e-3),
)
drive = net.stimulus([[0.2, 0.4, 0.1]])
phase = net.settle(drive)
assert np.all(phase.qualified)
answer = net.read(phase)  # output neurons inside the same graph
assert np.isclose(cd.certificate(net.brain).rate, 0.75)
lesson = net.observe(drive, [[0.3, 0.1]], source_id="example:recursive-1")
assert lesson.updated, lesson.reason
assert all(np.all(p.qualified) for p in (lesson.free, lesson.plus, lesson.minus))

wide = cd.PatchNet.recursive(1371, [24, 8], 6, coupling=1.8)
assert wide.brain.connectome.n == 1409 and wide.brain.connectome.synapses == 88512
assert wide.learner.parameters() == 45665
assert np.isclose(cd.certificate(wide.brain).rate, 0.95)
# Permuted widths give the same unweighted graph up to relabelling.
assert (cd.PatchNet.recursive(3, [4, 2], 2).brain.connectome.synapses
        == cd.PatchNet.recursive(3, [2, 4], 2).brain.connectome.synapses)
```

The default `solver="local"` admits a phase on the full equation residual; the opt-in
`solver="hybrid"` adds a curvature check and Newton refinement on a narrower set of models
([the hybrid solver](patchnet.md#the-hybrid-solver)). Act on an answer only when
`phase.qualified` holds and count a step only when `lesson.updated` holds.

## Inspect the mechanism explicitly

Four neurons make the shared equilibrium checkable by hand: two base neurons (one of them
the input port, one the output) and two observers, wired with `W[post, pre]`. The test
evaluates every neuron's equation independently, perturbs each side and watches the other
move. After a cut of the interface in both directions, the observer's input leaves the
base unchanged.

```python
import numpy as np
import cadence as cd

W = np.array([  # W[post, pre]: base neurons 0, 1; observers 2, 3
    [0.00, 0.12, 0.22, 0.00],
    [0.12, 0.00, -0.04, 0.18],
    [0.22, -0.04, 0.00, 0.12],
    [0.00, 0.18, 0.12, 0.00],
])
tolerance, beta = 1e-10, 1e-4

def make_patch(weights):
    post, pre = np.nonzero(weights)
    graph = cd.Connectome.from_synapses(
        4, pre=pre, post=post, sign=weights[post, pre],
        populations={"input": [0], "output": [1], "base": [0, 1], "observer": [2, 3]},
    )
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model(leak=1.0),
                           bias=np.full(4, 0.15), backend="cpu")
    config = cd.LearnerConfig(nudge="quadratic", beta=beta, eta=0.01, eta_bias=0.01)
    return cd.PatchNet(cd.Learner(brain, [1], config), steps=1024, chunk=8,
                       tolerance=tolerance)

def qualify(phase, weights, drive, strength=0.0):
    # Residual of v = W tanh(v/2) + drive + bias (+ nudge) for every neuron.
    v = phase.state.v
    activity = np.tanh(v / 2)
    defect = activity @ weights.T + drive + 0.15 - v
    defect[:, 1] += strength * (0.35 - activity[:, 1])
    assert np.all(phase.converged) and np.max(np.abs(defect)) <= tolerance
    return activity

def solve(patch, weights, drive):
    patch.reset()
    return qualify(patch.settle(drive), weights, drive)

patch = make_patch(W)
drive = np.array([[0.4, 0.0, 0.0, 0.0]])
free = solve(patch, W, drive)
observer_input = drive + [[0.0, 0.0, 0.0, 0.4]]
assert np.max(np.abs(solve(patch, W, observer_input)[:, :2] - free[:, :2])) > 1e-3
base_input = drive + [[0.4, 0.0, 0.0, 0.0]]
assert np.max(np.abs(solve(patch, W, base_input)[:, 2:] - free[:, 2:])) > 1e-3

cut = W.copy()  # reciprocal cut: both directions, every neuron kept
cut[:2, 2:] = cut[2:, :2] = 0
control = make_patch(cut)
np.testing.assert_allclose(solve(control, cut, observer_input)[:, :2],
                           solve(control, cut, drive)[:, :2], atol=1e-10, rtol=0)

patch.reset()  # all three teaching phases solve the same combined graph
lesson = patch.observe(drive, [[0.35]], source_id="example:teaching-1")
assert lesson.updated
for phase, strength in [(lesson.free, 0), (lesson.plus, beta), (lesson.minus, -beta)]:
    qualify(phase, W, drive, strength)  # pre-update W and bias
np.testing.assert_array_equal(patch.state.v, lesson.free.state.v)
```

Here the free map's Lipschitz bound is `0.5 * max(sum(abs(W), axis=1)) = 0.19`, and the
small nudges keep the map a contraction. An arbitrary graph needs its own stability
analysis; a small measured residual does not show uniqueness or a minimum.

## Compare depth fairly

Layer labels describe construction. Hidden populations of equal width are structurally
interchangeable and permuted widths give the same unweighted graph, so only learned
weights tell levels apart; the same seed draws different weights after a relabelling.
The deepest observer is changed by the levels it reads, so it holds no authority over
them, and a narrow observer can lose information held elsewhere. Adding a level changes
width, parameters and contacts together. A depth claim needs these controls at matched
parameters, contacts, information, experience and total work, failed solves included:

| Control | Built as | Tests |
| --- | --- | --- |
| Widening | a shallow `[n]` with the same parameter or contact count | whether depth beats width |
| Cut feedback | zero both directions of the observer interface, as above | whether observer feedback carries the effect |
| Frozen readback | hold the observer's readback fixed across the solve | whether the observer must settle with the base |
| Sequential observer | a separate observer run after the base settles | whether one shared equilibrium matters |

A one-way cut leaves contacts without a reciprocal partner; `PatchNet` refuses it.

For comparisons with transformers or other families, follow the
[checklist](evaluation.md#comparing-with-other-model-families). Deep equilibrium models
apply fixed-point solves to attention architectures
([Bai, Kolter and Koltun, 2019](https://arxiv.org/abs/1909.01377)) and are a natural
comparison family; an efficiency claim needs measured work to the same held-out target.
