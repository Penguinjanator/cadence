# The settling certificate

`cd.certificate(brain)` gives a sufficient condition for a brain's free settling to converge
to exactly one equilibrium, and bounds the distance to it after a finite number of steps. It
costs one pass over the synapses.

## The rule

One settling step of a brain without adaptation applies the free map

```text
F_c(v) = (1 - dt) v + dt (W act(v) + c)
```

`W` holds the effective signed weights, contact counts and gains included; `c` holds the bias
and the current sensory, trace and associative-memory drive. Let `L` be the largest slope of
the activation (`cd.lipschitz_constant(model)`) and `rho` the largest absolute incoming
effective weight sum of any neuron (`cd.row_mass(brain)`). If `L rho < 1` and `0 < dt <= 1`,
each step shrinks the largest potential difference between two states by
`q = 1 - dt (1 - L rho) < 1`. By the Banach fixed-point theorem there is exactly one
equilibrium, and

- after `k` steps from any start the distance to it is at most `q^k / (1 - q)` times the first
  step's movement, and the last step's movement `m` bounds it by `m / (1 - q)`;
- after the drive changes by `delta` (largest change of any drive), a warm start from the old
  equilibrium is within `tol` of the new one after about
  `log(dt delta / ((1 - q) tol)) / log(1 / q)` steps, and needs none after no change;
- with no stimulus and no bias the equilibrium is rest;
- removing neurons moves the equilibrium by at most `dt` times the synaptic mass they sent,
  times the activation bound, over `1 - q`.

For the library's activation `L = (slope / 4) max(1 / (1 - rest), leak / rest)`, with `rest`
the rest emission. Under `learning_neuron_model()` (slope 1, threshold 0, leak 0.1) `L = 1/2`,
so a learning brain is certified when every incoming effective weight sum is below 2.
`certificate` raises `ValueError` for a model whose `dt` lies outside (0, 1].

## Reading it

```python
import cadence as cd

connectome = cd.layered(8, 12, 3, density=1.0, init=0.25)   # row mass 1.39: certified
brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
cert = cd.certificate(brain)
print(cert.rate, cert.certified, cert.mass_limit)   # q, the verdict, and 1 / L
print(cert.error_bound(1e-4))                       # distance left after a step that moved 1e-4
print(cert.apriori_bound(1.0, steps=50))            # after 50 steps, first movement 1.0
print(cert.steps_for(change=0.5, tolerance=1e-3))   # warm-start budget after a drive change
print(cert.to_dict())                               # fields, rate, verdict and limit
```

| Name | Meaning |
| --- | --- |
| `row_mass`, `lipschitz`, `dt` | `rho`, `L` and the model's integration step |
| `adaptation`, `certified` | whether the model adapts; `certified` is `rate < 1` without adaptation |
| `rate`, `mass_limit` | the factor `q`; `1 / L`, the row mass below which the model is certified |
| `error_bound(movement)` | `movement / (1 - q)` for each entry; `inf` when uncertified |
| `apriori_bound(first_movement, steps)` | `first_movement q^steps / (1 - q)`; `inf` when uncertified |
| `steps_for(change, tolerance)` | warm-start steps; 0 for no change, at most 1 at rate 0; `ValueError` when uncertified |

`error_bound` takes potential movement. `NeuralGraph.residual` reports the equation error
before multiplication by the step, so pass `brain.neuron_model.dt * brain.residual(drive,
state)` (unmasked free phase). The potential bound times `cert.lipschitz` bounds activation
error. `settle_batch(tolerance=...)` measures activation movement and must not be substituted.
The warm-start budget assumes the old state is the old equilibrium; an approximate warm state
adds its own remaining error.

The certificate covers the free phase without adaptation. A nudge adds a drive the argument
omits, adaptation adds a slow variable, and records are read and written without settling. For
those, and above the limit, `residual` and `equilibrate` qualify the equations at the measured
state and say nothing about uniqueness or convergence. A failed certificate means only that
this sufficient argument does not apply.

## Using it

- Report `cert.to_dict()` in every receipt next to the residual.
- Recompute `certificate(brain)` after learning.
- Keep learned brains under the limit where you can: bound efficacies, scale the gain by the
  fan-in and watch `row_mass` over a life. `LearnerConfig.scale_cap` (default 8) clips each
  plastic efficacy separately, which does not bound a row sum.
- Warm-start streams: the step bound is logarithmic in the change. A certified brain may need
  more steps than a given finite budget allows.
- Lesions keep the certificate: ablation only removes synapses and lowers the row mass.

## Equilibrium-propagation scope

`cd.ep_structure(brain, *, fixed_inputs=(), tolerance=1e-12)` returns an `EPStructure`, a
structural check that the graph admits the symmetric free-state argument of
[equilibrium propagation](https://arxiv.org/abs/1602.05179). It is separate from the
contraction certificate. Repeated or out-of-range `fixed_inputs` and a negative or nonfinite
`tolerance` raise `ValueError`.

| Field | Meaning |
| --- | --- |
| `fixed_inputs`, `free_neurons` | the declared inputs and the count of the other neurons |
| `free_asymmetry` | largest gap between a free/free effective weight and its reverse (parallel edges summed) |
| `fixed_incoming_mass` | largest incoming effective mass into a declared input; nonzero means it is no source |
| `adaptation`, `tolerance` | adaptation present; the tolerance of both checks |
| `compatible` | both gaps within `tolerance` and no adaptation; `to_dict()` includes it |

Source inputs settle to potentials set by their fixed drive and bias, and their projections act
as external fields on the free neurons without reverse edges. Keep them unchanged and unnudged
between phases. Efficacy tying alone does not give effective symmetry when contact counts or
gains differ. A compatible structure alone does not make the contrast a gradient: phases must
converge on a smooth stable branch, and finite beta keeps an estimation bias. Convert the raw
contrast as in [learning](learning.md#5-why-the-contrast-is-a-gradient);
[`tests/test_equilibrium.py`](../tests/test_equilibrium.py) checks it against finite differences
and [`tests/test_runtime_checks.py`](../tests/test_runtime_checks.py) the refusals.

## Related results and their assumptions

Keep five questions separate: does an equilibrium exist, is it unique, does this iteration reach
it, does the continuing life preserve useful context, does its behaviour succeed. Neither the
certificate nor residual qualification answers the last two. Agreement means satisfying
the `F_c` relations; neurons need not share an activity. Check each result below against the
executed update before reusing it.

[Dijkstra (1974)](https://www.cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD426.html) defines
self-stabilization: legitimate states, moves that preserve them, and specific local algorithms
reaching them from every start under a central daemon. A recovery test reuses the pattern by
naming the disturbance, the recovery condition and the continuation it must survive. The
Cadence life has no arbitrary-state self-stabilization guarantee.

[Bertsekas (1983), sections 3 to 4.2](https://web.mit.edu/dimitrib/www/Distr_Comp_Fixed.pdf)
proves asynchronous convergence for a map contracting in a weighted maximum norm, if every
coordinate keeps updating and stale reads eventually expire. The row-mass certificate is such a
maximum-norm contraction for the fixed free map; Cadence settles synchronously.

[Simonetto (2017), Theorem 3.1(b)](https://arxiv.org/abs/1704.07338) tracks a moving fixed
point: for maps contracting by `q` on a common domain and norm, with iterate error `eta` and
drift `delta` per step, `e[t] <= q^t e[0] + (eta + delta) (1 - q^t) / (1 - q)`. Observation,
trace, recalled records and weights all move Cadence's target.
[Bernstein and Dall'Anese (2018)](https://arxiv.org/abs/1804.09768) add delays and drops.

[Hansen and Ghrist (2019)](https://arxiv.org/abs/1808.01513) define the sheaf Laplacian
`L = D* D` for fixed restriction maps; its kernel holds the compatible assignments, whose local
values may differ. [Their 2021 paper](https://arxiv.org/abs/2005.12798) shows that diffusion
`dx/dt = -alpha L x` converges to the projection onto `ker L`, and a held boundary gives the
nearest harmonic extension. It is a symmetric comparison model, distinct from Cadence's directed
nonlinear operator.

[Hanks et al. (2025)](https://arxiv.org/abs/2504.02049v2) extend the flow to strongly convex
edge potentials, `dx/dt = -alpha D* grad U(D x)`, and
[Zhao et al. (2025)](https://arxiv.org/abs/2510.00270v1) to asynchronous updates with bounded
stale reads and a small step. Cadence's activation, directed weights, nudges and memories lack
this gradient form; the gradient of a squared residual adds a Jacobian transpose.
