# The settling certificate

A settled state is only an answer if the settling was going somewhere. The certificate says
when it was, and how far the answer can be from the equilibrium after any number of steps.

## The rule

One settling step of a brain without adaptation moves every potential toward its synaptic
input plus stimulus:

```text
v <- (1 - dt) v + dt (W act(v) + c)
```

Let `L` be the largest slope of the activation and `rho` the largest absolute incoming
effective weight sum of any neuron (the row mass). If `L rho < 1` and `0 < dt <= 1`, one step
shrinks the largest potential difference between any two states by the factor

```text
q = 1 - dt (1 - L rho) < 1
```

Then there is exactly one equilibrium, and

- after `k` steps from any start, the distance to it is at most `q^k / (1 - q)` times the
  first step's movement;
- the last step's movement `m` bounds the remaining distance by `m / (1 - q)`;
- after the stimulus changes by `delta` (largest change of any drive), a warm start from the
  old equilibrium is within `tol` of the new one after about
  `log(dt delta / ((1 - q) tol)) / log(1 / q)` steps, and after no change it is there at once;
- with no stimulus and no bias the equilibrium is rest, exactly;
- removing a set of neurons moves the equilibrium by at most `dt` times the synaptic mass
  they sent, times the activation bound, over `1 - q`.

For the library's activation the slope bound is `(slope / 4) max(1 / (1 - rest), leak / rest)`
with `rest` the rest emission. Under `learning_neuron_model()` (slope 1, threshold 0, leak
0.1) that is `1/2`, so a learning brain is certified when every neuron's absolute incoming
effective weight sum is below 2. These statements are theorems with Lean proofs in the
flagship paper's library.

## Reading it

```python
import cadence as cd

connectome = cd.layered(8, 12, 3, density=1.0)
brain = cd.Brain(connectome, cd.learning_neuron_model())
cert = cd.certificate(brain)
print(cert.row_mass, cert.lipschitz, cert.rate, cert.certified)
print(cert.mass_limit)                 # the row mass below which this model is certified
print(cert.error_bound(1e-4))          # remaining distance after a step that moved 1e-4
print(cert.steps_for(change=0.5, tolerance=1e-3))   # warm-start budget after a change
```

`cd.row_mass(brain)` and `cd.lipschitz_constant(model)` are the two ingredients. The
certificate covers the free phase without adaptation; a nudge adds a drive the argument does
not include, and adaptation adds a slow variable. For those, and for any brain whose row mass
is above the limit, `Brain.residual` and `Brain.equilibrate` remain the checks: they certify
the equations at the state they measure, and nothing about uniqueness or convergence.

## Using it

- Report `cert.to_dict()` in every receipt next to the residual.
- Keep learned brains under the limit where you can: cap efficacies (`SCALE_CAP`), scale the
  gain by the fan-in, and watch `row_mass` over training. A brain that crosses the limit may
  settle in practice; the certificate then says nothing.
- Warm-start streams. The step bound is logarithmic in the change, and an unchanged input
  costs no step at all.
- Lesions keep the certificate: cutting synapses can only lower the row mass.
