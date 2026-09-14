# Small components, tested separately

The biological analogy motivates experiments; it does not establish that Cadence
implements an animal brain. Its observer-like software patches expose local state,
ports, readback, records, and repair. `tests/test_child.py` checks a few small
compositions before they are used inside larger systems.

| component | controlled task | observed boundary |
|---|---|---|
| a trace of changing input | read a symbol from the preceding moment against a static background | focused input trace succeeds in this toy; a hidden-state trace need not |
| an eligibility trace | credit a press paid three moments later | the trace helps on this short bandit; it does not solve arbitrary delayed credit |
| a reward baseline and quiet band | repeated reward followed by a surprise | usual reward can produce no update; the band and scale are supplied |
| capped warm settlement | compare state with the preceding input's state | a transient retains history; this is not an equilibrium or an addressed-memory guarantee |
| salience-weighted eligibility | delayed cue with static background | it does not consistently improve over the plain trace in the tested toy |

The tests isolate mechanisms, not broad capabilities such as permanent concepts,
curiosity, or general game playing. Slow weights can forget; a fading trace can lose
an event; high-gain dynamics can saturate. Episode resets and observation/reward order
are part of the experiment contract.

For explicit revisable records, [residual fast memory](memory.md) is a smaller route:
read, compare with the observation, and write only the error. Its independent LMS,
interference, and reset tests live in `tests/test_fast_memory.py`. The public
[changing-memory example](https://github.com/muellerberndt/cadence-examples/tree/7302f2af3dc0638bbafd1da1446ed96ffabaa9dd/05_memory)
adds exact lookup controls and a trained transformer comparison.
