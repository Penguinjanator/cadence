# A trained brain in a browser page

For inference without adaptation, a page can store the effective weight matrix,
the bias vector and the neuron model's activation, and settle the brain locally. The
implementation below starts each input from rest and omits masks, nudges and
retained memory.

## Export

Continue with the learner from the [quickstart](quickstart.md#learn-a-response):

```python
brain = learner.brain
connectome = brain.connectome
neuron_model = brain.neuron_model
assert neuron_model.adaptation is None
payload = {
    "n": connectome.n,
    "populations": {name: [int(i) for i in members] for name, members in connectome.populations.items()},
    "W": brain.dense().ravel().tolist(),                    # W[pre, post], row-major
    "bias": brain.bias.tolist(),
    "neuron": {"slope": neuron_model.slope, "threshold": neuron_model.threshold,
               "leak": neuron_model.leak, "dt": neuron_model.dt,
               "stimulus": neuron_model.stimulus_amplitude, "rest": neuron_model.rest_emission},
}
```

`Brain.dense()` returns the synapse matrix with every effective drive folded in
(`gain`, `count`, `efficacy`, `log_gain`). Serialize `payload` as JSON and load
it in the page. The dense matrix stores `n * n` values, even for a sparse connectome;
check payload size before embedding a large model. Rounding weights changes the
model and requires another numerical comparison.

## Settle

After loading that JSON as `payload`, this implements the declared inference
subset. `drive` holds one drive value per neuron, already scaled by the stimulus
amplitude:

```javascript
const n = payload.n, W = payload.W, BIAS = payload.bias, NEURON = payload.neuron;
function act(v) {
  const r = 1 / (1 + Math.exp(-NEURON.slope * (v - NEURON.threshold))) - NEURON.rest;
  return r > 0 ? r / (1 - NEURON.rest) : NEURON.leak * r / NEURON.rest;
}
function settle(drive) {                       // drive: one number per neuron, the stimulus
  const v = new Float64Array(n), s = new Float64Array(n), synapticInput = new Float64Array(n);
  for (let t = 0; t < 100; t++) {
    synapticInput.fill(0);
    for (let i = 0; i < n; i++) { const si = s[i]; if (si === 0) continue;
      for (let j = 0; j < n; j++) synapticInput[j] += si * W[i * n + j]; }
    let moved = 0;
    for (let j = 0; j < n; j++) {
      v[j] += NEURON.dt * (-v[j] + synapticInput[j] + drive[j] + BIAS[j]);
      const ns = act(v[j]); moved = Math.max(moved, Math.abs(ns - s[j])); s[j] = ns;
    }
    if (moved < 1e-4) break;
  }
  return s;
}
```

Compare its outputs against `brain.settle` on the CPU backend with `steps=100` and
`tolerance=1e-4` on representative drives. JavaScript uses double-precision numbers,
but operation ordering can change rounding. Measure latency in the target browser; an
activation stopping tolerance alone does not certify an equilibrium.

## Show the settling

Record the output neurons' activations at every step and replay them as bars over a few
animation frames, so the user sees the answer form. Show the hidden neurons at rest as a
strip of squares with opacity as activation. Print the number of steps taken. The
[cadence-examples](https://github.com/muellerberndt/cadence-examples) websites
separate small numerical kernels from rendering and check them against Python.
Their `shared/shell.html` and view modules are starting points, and
`python tools/build_showcase.py` regenerates the six websites and the gallery.
The [hosted versions](https://floatingpragma.io/cadence-examples/) serve the same files.
