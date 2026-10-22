# Retained admission failures

`admission-256.json` is the first declared recipe; `admission-1024.json` is its
solver-budget retry. `damping-readback.json` is a diagnostic replay of the latter
with the existing fallback, with its extra work recorded separately. None
admits a behavioral coupled candidate. See the [experiment guide](../README.md)
for the scores, boundaries and pending gates.

The `sources/` text files preserve exact historical source bytes by SHA-256.
They are evidence, not imported implementations. The verifier maps a historical
manifest entry to its retained bytes before hashing, and uses unchanged source
files for the remaining entries. To reproduce an earlier runner, restore those
bytes at their manifest's logical paths in a disposable checkout of baseline
`ce386af2f5d50bb4d6a275d9a3e686a50f3fc6fa`, then run its declared protocol.
Do not restore them over the current experiment or alter a receipt to match it.

Source/digest/arithmetic verification checks custody and reported decisions;
it does not establish behavioral reproduction, stability or useful relevance.
