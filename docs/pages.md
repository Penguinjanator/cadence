# The brain in a browser page

Every page that shows a Cadence brain shows the whole brain: every neuron a point coloured
by its region, every synapse a line, laid out so that connected neurons sit near each other,
with the activity and the change of every settling step animated live and the traces of
every region below. One component, shipped with the library, draws it, and any recording
replays in any page.

## The atlas

`build_atlas` lays a connectome out; `atlas_of(brain)` uses the brain's effective weights.
Regions are the connectome's populations (coarsest first; a `regions=` map overrides them).
Their role and colour follow their name: vision, sensory, memory, association, motor, value
(`roles=` overrides). Regions are placed by a force layout of the region graph, so strongly
connected regions sit next to each other. Inside a region, neurons are placed by their
synapses (iterated neighbour averaging, a spectral-style embedding), by a declared sheet
shape (`shapes={"retina": (rows, cols)}` gives a grid), or by supplied coordinates
(`positions={"cortex": array}`), or every neuron by one shared anatomical frame
(`positions={"*": array}` with a row per neuron, kept as given). Everything is deterministic
under `seed`.

```python
import cadence as cd

atlas = cd.atlas_of(brain, shapes={"retina": (32, 32)})
print(atlas.summary())          # neurons, synapses, regions with roles
payload = atlas.to_json()       # positions, regions, every synapse, base64 arrays
```

`atlas.subsample_edges(limit, seed=None)` keeps at most `limit` synapses, drawn with
probability proportional to their absolute weight, when a page cannot draw them all.

## The renderer

`cd.brain_scan_script()` is the source of `brain_scan.js`, an ES module with no
dependencies. Inline it in a page (drop the `export` keywords) or serve it as a file.

```javascript
const scan = new BrainScan(canvas, ATLAS, { labels: labelsDiv, strip: stripCanvas });
scan.reset(activation);      // a new stimulus: the next step measures change from here
scan.step(activation);       // after every settling step: activation per neuron ({draw: false} defers the frame)
scan.set(activation);        // show a state without measuring change
scan.show(activation, heat, { level });  // a page's own signals: messages, glow, brightness
scan.setWeights(weights);    // the synapses learned; scan.setVisible(mask) hides lesioned neurons
scan.draw(now);              // every animation frame between steps: the messages travel
scan.onhover = (hit) => ...; // {neuron, region, activation, change, heat, potential}
scan.fit(); scan.snapshot(); // {renderer, neurons, synapses, steps, zoom, allEdgesSubmitted}
scan.screen(i); scan.toScreen(x, y);     // CSS pixels, for a page's own overlay
playFrames(scan, FRAMES, { fps: 30 });   // replay quantised recorded steps
```

What it draws, as a scan: tissue in each region's colour whose brightness is the activation
(the field of every neuron, so a region reads as one glowing organ); a hot glow where neurons
changed in the last steps, on a scan colour map from violet through magenta and orange to
white, fading with `heatDecay`; synapses that light up when their presynaptic neuron just
changed; particles travelling along synapses in proportion to the message sent, so a
settling is visible as a wave through the connectome; region labels; and a montage strip
with one EEG-style row per region (its change as a line, its activity as a fill) and the
whole brain on top. `mode` selects the brightness: `activity`, `potential` or `change`.
Brains with more synapses than `particleBudget` (300,000) draw particles for a uniform
sample of them; every synapse is still rasterised. Scroll zooms, drag pans, hover inspects.
WebGL2 draws it; without it, a Canvas2D fallback draws the neurons.

The renderer draws the connectome, the settled regions. The example pages draw the records
cortex beside the scan with their own component, `records_view.js`: the granule raster with
the active cells of the executed reading, imagined reads, writes scaled by the record rate,
the habituated reading and the per-field reads.

`layoutAtlas({ n, pre, post, weight, groups, shapes, positions, roles, labels, seed })` is the
same layout in the browser, for pages that build their brains at run time: `groups` names a
region per neuron, `shapes` declares sheets, `positions` supplies coordinates per region or,
under the key `"*"`, one shared anatomical frame for every neuron. `scan.setAtlas(atlas)`
loads a new brain into the same canvas.

## One call for a page

`atlas.page(*, frames=None, brain=None, title="Cadence brain scan", note=..., inputs=None, limit=2000)`
returns a self-contained HTML page: the atlas, the renderer, and either recorded frames or
a live brain, or both. `note` is the text under the title; `inputs` names the neurons that
the live page's `Detune` drives, by default the first population whose name reads as
vision or sensory, or else the first eight neurons.

```python
records = []
with cd.record_settlements(records.append, label="probe"):
    brain.settle_batch(drive, steps=60, tolerance=None)
frames = atlas.frames_from_record(records[0])            # eight bits per neuron and step
html = atlas.page(frames=frames, brain=brain, title="A brain settles")
```

With `brain=`, the page embeds the dense weights and settles in the browser: `Detune` draws
a new stimulus on the input region and the brain settles into its new equilibrium step by
step, at the chosen rate, with the certificate's error bound in the status line when the
brain is certified. The dense matrix is `n * n` numbers, so the live option is capped at
2,000 neurons (`limit=` raises it). Larger brains use recorded frames or their page's own
engine. Write the returned `html` with `Path("brain.html").write_text(html, encoding="utf-8")`.

## A page with its own engine

For inference without adaptation, a page can store the effective weight matrix, the bias
vector and the neuron model's activation, and settle the brain locally, then feed every
step to the scan. `Brain.dense()` returns the synapse matrix with every effective drive
folded in (`gain`, `count`, `efficacy`, `log_gain`).

```python
payload = {
    "n": connectome.n,
    "W": brain.dense().ravel().tolist(),   # W[pre, post], row-major
    "bias": brain.bias.tolist(),
    "neuron": {"slope": neuron_model.slope, "threshold": neuron_model.threshold,
               "leak": neuron_model.leak, "dt": neuron_model.dt,
               "stimulus": neuron_model.stimulus_amplitude, "rest": neuron_model.rest_emission},
}
```

```javascript
const n = payload.n, W = payload.W, BIAS = payload.bias, NEURON = payload.neuron;
function act(v) {
  const r = 1 / (1 + Math.exp(-NEURON.slope * (v - NEURON.threshold))) - NEURON.rest;
  return r > 0 ? r / (1 - NEURON.rest) : NEURON.leak * r / NEURON.rest;
}
function settle(drive, scan) {              // drive: one number per neuron, the stimulus
  const v = new Float64Array(n), s = new Float64Array(n), synapticInput = new Float64Array(n);
  scan.reset(s);
  for (let t = 0; t < 100; t++) {
    synapticInput.fill(0);
    for (let i = 0; i < n; i++) { const si = s[i]; if (si === 0) continue;
      for (let j = 0; j < n; j++) synapticInput[j] += si * W[i * n + j]; }
    let moved = 0;
    for (let j = 0; j < n; j++) {
      v[j] += NEURON.dt * (-v[j] + synapticInput[j] + drive[j] + BIAS[j]);
      const ns = act(v[j]); moved = Math.max(moved, Math.abs(ns - s[j])); s[j] = ns;
    }
    scan.step(s);
    if (moved < 1e-4) break;
  }
  return s;
}
```

Compare its outputs against `brain.settle` on the CPU backend with `steps=100` and
`tolerance=1e-4` on representative drives. JavaScript uses double-precision numbers, but
operation ordering can change rounding. An activation stopping tolerance alone does not
certify an equilibrium; the [certificate](certificate.md) does, and a page can show its
bound from the last movement. Keep a website's numerical kernel separate from its
rendering so the same recorded observations can check it against Python.
