# A trained net in a browser page

For inference without adaptation, a page can store the effective weight matrix,
bias vector, and activation rule and run settlement locally. The implementation
below starts each input from rest and omits masks, nudges, and retained memory.

## Export

Continue with the learner from the [quickstart](quickstart.md#learn-a-response):

```python
engine = learner.engine
wiring = engine.wiring
rule = engine.rule
assert rule.adaptation is None
payload = {
    "n": wiring.n,
    "sets": {name: list(members) for name, members in wiring.sets.items()},
    "W": engine.dense().ravel().tolist(),                   # W[pre, post], row-major
    "bias": engine.bias.tolist(),
    "rule": {"slope": rule.slope, "threshold": rule.threshold, "leak": rule.leak,
             "dt": rule.dt, "clamp": rule.clamp_amplitude, "rest": rule.rest_emission},
}
```

`Settlement.dense()` returns the overlap matrix with every effective drive folded in
(`gain`, `count`, `edge_scale`, `log_gain`). Serialize `payload` as JSON and load
it in the page. The dense matrix stores `n * n` values, even for sparse wiring;
check payload size before embedding a large model. Rounding weights changes the
model and requires another numerical comparison.

## Settle

After loading that JSON as `payload`, this implements the declared inference
subset. `drive` contains full-owner drive values, already scaled:

```javascript
const n = payload.n, W = payload.W, BIAS = payload.bias, R = payload.rule;
function act(v) {
  const r = 1 / (1 + Math.exp(-R.slope * (v - R.threshold))) - R.rest;
  return r > 0 ? r / (1 - R.rest) : R.leak * r / R.rest;
}
function settle(drive) {                       // drive: one number per owner, the clamp
  const v = new Float64Array(n), s = new Float64Array(n), inbox = new Float64Array(n);
  for (let t = 0; t < 100; t++) {
    inbox.fill(0);
    for (let i = 0; i < n; i++) { const si = s[i]; if (si === 0) continue;
      for (let j = 0; j < n; j++) inbox[j] += si * W[i * n + j]; }
    let moved = 0;
    for (let j = 0; j < n; j++) {
      v[j] += R.dt * (-v[j] + inbox[j] + drive[j] + BIAS[j]);
      const ns = act(v[j]); moved = Math.max(moved, Math.abs(ns - s[j])); s[j] = ns;
    }
    if (moved < 1e-4) break;
  }
  return s;
}
```

Compare its outputs against CPU settlement with `steps=100` and `tolerance=1e-4`
on representative drives. JavaScript uses double-precision numbers, but operation
ordering can change rounding. Measure latency in the target browser; an
activation stopping tolerance alone does not certify an equilibrium.

## Show the settlement

Record the output owners' activations at every step and replay them as bars over a few
animation frames: the user sees the answer form. Show the hidden owners at rest as a strip
of squares with opacity as activation. Print the number of steps taken. The current [browser showcase](https://github.com/muellerberndt/cadence-examples/tree/main/showcase)
separates small numerical kernels from rendering and checks them against Python.
Use its `shell.html` and view modules as starting points. Run
`python tools/build_showcase.py` to regenerate the two entry pages; launch each
website with `python serve.py mouse`, `eye-arm`, `fly`, `worm`, or `memory`.
