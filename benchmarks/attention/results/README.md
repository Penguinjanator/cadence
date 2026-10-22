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

## Acquisition repair

`repair-development-precheck.json` retains the first matched-policy development
run and its exact original runner under `sources/001bbc37*.txt`.
`repair-development.json` repeats it after tightening only `gate()` and `check()`;
every per-run reading, parameter/checkpoint digest and work counter is identical.
`repair-confirmation.json` binds that strict development receipt and records all
twenty previously unused founders under the selected, unchanged recipe.

Both development wrong-row lives and all 20 confirmation lives reach 100% on
overall, unequal-fact and each-query accuracy with complete training and no
refusals. The every-row controls still fail and remain in the development
receipts. This passes the declared exposed acquisition control; it does not
establish recalled-content use, learned selection or semantic generalization.
See the [repair protocol](../repair-protocol.json) and [results](../README.md#acquisition-repair-the-existing-wrong-answer-teaching-control).
