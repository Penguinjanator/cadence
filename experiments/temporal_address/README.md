# A causal sensory delay on the actual repeat-card task

This experiment runs POPGym 1.0.7 `RepeatPreviousEasy` with its original four
suits, `k=4`, 51 transitions per episode and 48 rewarded decisions. The installed
environment scores `hand[-4]` before dealing its next card: the correct suit is
three observations behind the current one. The reward remains ±1/48 and total
return remains in [-1, 1]. Only training multiplies rewards by 48, as in the
existing observation-alignment diagnosis.

The memory is a **controller-owned NumPy register outside the joint neural
solve**, with three consecutive one-step stages and four coordinates per stage.
It starts empty each episode. A decision reads the oldest stage, receives the
current observation, acts, observes reward and learns; only then does the
register shift in that actual current observation. It receives no clock,
position, suit-to-action label, environment hidden state, or future target.
The fixed three-step lag is a task-specific architectural prior. It is neither
learned addressing nor a general mechanism for temporal credit.

Every policy observes the same eight coordinates: current suit and oldest
stored suit, each one-hot encoded. The masks zero only the latter four
coordinates. All patches use 32 hidden neurons. The direct-route patch adds
all-to-all projections from the eight sensory coordinates to four motors,
initialized at zero; it does not prewire matching suit/action identities. The
no-direct control retains the identical initial hidden-network weights. A
smaller conventional linear softmax score policy learns from the same sampled
actions and scalar rewards. It receives no supervised action labels.
Including each critic, the direct patch has 499 trainable numbers, the no-direct
patch 467, and the linear policy 45. The 12 transient register coordinates per
environment are separate from those learned parameters.

The patch uses the corrected `ActorCritic` from frozen integration commit
`2de0383`. Its actor, bias and critic rates are 0.02, 0.002 and 0.1; gamma and
lambda are zero, matching the immediate-reward alignment diagnosis. The critic
fits raw TD targets. Each decision clears transient neural/eligibility state,
so the designed sensory register is the only inter-decision memory. Learned
parameters and the action RNG persist. The conventional linear control uses
the categorical log-policy score, the same three rates, a fitted linear critic,
and reward modulation clipped to [-1, 1]. Neither algorithm receives future
reward credit.

These are bounded observer-like software patches: explicit local state and
read/write ports, reward feedback, recorded observations and actions, and
public evidence receipts. The register is an explicit implementation choice,
not an emergent property of the neural dynamics.

## Schedule and evidence

The seed-91 pilot ran five vector episodes of 32 environments per arm. Its
returns were 0.113333 (direct-memory patch), -0.497917 (masked patch), -0.378750
(no-direct patch), 1.0 (linear with memory), and -0.487500 (masked linear).
The incomplete learning outcomes remain in `runs/pilot/receipt.json`.

The fixed confirmation uses seeds 0–4 and 200 vector episodes of 32 environments
per arm, followed by 100 fresh evaluation episodes. This is **326,400 training
interactions per arm and seed**, exactly the per-arm budget of the existing
`t1_alignment/receipt_200.json` and refactor receipt. The original
`t1_repeat_previous/receipt_easy.json` used 300 vector episodes, or 489,600
training interactions per arm; this experiment uses two thirds of that budget.
Each new arm/seed adds 5,100 held-out interactions. The confirmation totals
8,160,000 training plus 127,500 evaluation interactions; the pilot separately
totals 66,300 interactions. No hyperparameter sweep or test-based selection is
performed. The new seeds share each arm's environment reset schedule.

The [summary receipt](runs/summary/receipt.json) records every paired difference
and these held-out means. SD is the sample standard deviation across five
initializations and environment seed sets; each seed evaluates 100 episodes.

| Arm | Mean return | Sample SD |
| --- | ---: | ---: |
| Direct-memory patch | 1.000000000 | 0.000000000 |
| Masked direct patch | -0.491833333 | 0.009277736 |
| Memory patch without direct projections | 1.000000000 | 0.000000000 |
| Linear policy with the same memory | 1.000000000 | 0.000000000 |
| Masked linear policy | -0.492416667 | 0.010057992 |

All five seeds solve the held-out task in each memory-equipped arm. The direct
patch's paired gain over its mask is 1.491833333 (SD 0.009277736); its difference
from both the no-direct patch and linear policy is zero in every seed. Thus the
fixed-budget result does not support necessity of the added projections or an
advantage over conventional learning. The task is learnable with this explicit
sensory-history representation and the corrected learner. Its memory-masked
controls remain near the random-action expected return of -0.5.

The main schedule took 698.167887 measured seconds over its 25 arm/seed runs;
the five pilot arms separately took 2.860117 seconds. All outcomes, including
the pilot's unsuccessful patch policies, are retained.

The original task and alignment source files and historical Easy receipt are
copied under `original/` and hashed in each new receipt. `core/` contains the
complete recursive Python package from `2de0383`; `environment/popgym/` contains
the installed environment and deck implementation. Their licenses are supplied.
Each run freezes all numerical sources again under its own `source/` directory.
Later formatting-only runner edits do not replace those executed bytes.
NumPy, Numba, Gymnasium and POPGym versions are recorded in `versions.json`.

Every held-out observation, action, received reward and memory read is retained
in a compressed artifact, along with learned parameters and every training
episode's return. Verification reconstructs the actual environment reward from
the observed history and action, checks the three-step memory relation, checks
artifact/source hashes and requires every scheduled arm and seed. These checks
do not rerun training or certify model correctness merely from stored losses.

## Reproduction

POPGym is an experiment-only dependency. From the repository root, use an
environment with the versions in `versions.json` and run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python experiments/temporal_address/run.py --output /tmp/cadence-temporal-reproduction
python experiments/temporal_address/run.py --verify experiments/temporal_address/runs/main/receipt.json
python experiments/temporal_address/run.py --verify experiments/temporal_address/runs/pilot/receipt.json
python experiments/temporal_address/summarize.py --verify
python -m unittest discover -s experiments/temporal_address -p 'test_*.py' -v
```

Choose a fresh output directory; the runner refuses to overwrite a nonempty
one. It imports its frozen core and POPGym implementation explicitly. The two
pure scheduling tests require only NumPy and can run without POPGym. CPU work
uses one BLAS thread; measured arm times include training and evaluation, with
initial Numba compilation charged to the first arm.
They are wall times on a shared machine, not controlled efficiency or energy
measurements.

This is a same-task representation and wiring diagnosis. The corrected learning
rule, memory representation and topology differ from the old runs, so an old/new
return change cannot be attributed to just one factor. The historical windowed
PPO already solved Easy; this experiment makes no advantage or exclusivity claim
against it.
