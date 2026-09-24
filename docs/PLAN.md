# The plan: every feature, its demo, and where it stands

> **Final goal.** A patch-net equilibrium brain that is more scalable, more capable and more efficient than a transformer, climbing the [metacognition ladder](METACOGNITION_LADDER.md) to a robot that acts and speaks among people.

This is the one list. The [ladder](METACOGNITION_LADDER.md) states the science of each rung (what the steering patch reads and returns, the experiments, the control, the falsifier); the GitHub issues carry the same acceptance line per step; the demos live in the workspace's `demos/` folder and, once they run in the browser, in the [examples repository](https://github.com/muellerberndt/cadence-examples). This page is the index of all of it, kept current as steps are accepted.

## The principles, and what "un-designed" means in practice

The main hypothesis stays: a brain that settles in global equilibria. The building block stays as simple as possible, like in nature, and every part of the brain answers with a settled state of that one rule. Natural evolution is always preferred to design:

- A governor, a steering patch, a habit, a port: each is a patch of the same rule, settled, never a hand-written rule or a feed-forward add-on. The hand-written rule is the control on the page.
- Anything that looks designed (a threshold, a budget, a horizon, a port's band and width, a motor coding, a fitness price) is a gene in a dict genome under `evolve` with `genes`, and the hand-set value is the control. Selection prunes what does not pay; every such outcome is kept, including the ones where the pruning goes against the designer.
- Within a life only the patch rule learns (the detuning contrast, the nudge, the adjoint); across lives the genome evolves.
- A fitness reads only what the genome cannot reweight (the room's gamed surprise is the exhibit).

## The rule of acceptance

Every step is accepted by a demo. The page has the shared skeleton (the whole brain in the viewer, the world, the instrument strip, the plain-words account), a three-way switch (the step on; off, the brain below it; the hand-designed control) so the acceptance is visible on the page, and the person in the world (the mouse as the laser dot, the ventriloquist's hand, the hand that shuffles the cups). The step is accepted when the demo works with the switch on and fails with it off. The page's layout: the playfield and the brain side by side at the same height, the rung selected on load, the laser or the drag as the only interaction, one small brain-selector line, four instrument tiles, everything else in the collapsed section.

## The list

Status: **accepted** (the demo works with the switch on and fails with it off, on the numbers in its report), **in progress**, **open**. "Where" is the demo's directory in the workspace's `demos/` folder, and the live page when it runs in the browser.

### The substrate (rung 0)

| step | feature | the demo | status | where |
| --- | --- | --- | --- | --- |
| [#3](https://github.com/muellerberndt/cadence/issues/3) | the shared core: belief inference and private imagination | the shell game | accepted 2026-09-23 | `demos/shell-game/` |
| [#7](https://github.com/muellerberndt/cadence/issues/7) | one cortex that lives: the three signals, the governor | the room with the heater and the window (rung 1's first page) | accepted 2026-09-23 | `demos/room/` |
| [#8](https://github.com/muellerberndt/cadence/issues/8) | habits as patches | a reaching arm learns its muscle memory | accepted 2026-09-23 | `demos/arm/` |
| [#9](https://github.com/muellerberndt/cadence/issues/9) | cortices joined by ports (in the library as `record_ports`) | an eye and an ear watch one ball | accepted for the eye in the dark; the ear through the port is open | `demos/eye-and-ear/` |
| [#15](https://github.com/muellerberndt/cadence/issues/15) | instruments: the brain's signals on every page | the polygraph | open (the tiles and the strip exist on every page; the shared component and receipt schema do not) | |
| [#16](https://github.com/muellerberndt/cadence/issues/16) | the steering patch across cortices | the referee | open | |
| [#4](https://github.com/muellerberndt/cadence/issues/4) | the player: scene-dependent consequences, internal foresight | ghost balls | open | |
| [#5](https://github.com/muellerberndt/cadence/issues/5) | language: valid memory probabilities, relational depth | the bedtime story | open | |
| [#6](https://github.com/muellerberndt/cadence/issues/6) | Connect Four: learned transitions, cheaper skilled decisions | blindfold Connect Four | open | |
| [#10](https://github.com/muellerberndt/cadence/issues/10) | understanding: withheld combinations | the alien zoo | open | |
| [#12](https://github.com/muellerberndt/cadence/issues/12) | sleep across cortices | sleep on it | open | |
| [#13](https://github.com/muellerberndt/cadence/issues/13) | evolution: wiring across lives, sizes that scale | the breeder | open | |
| [#14](https://github.com/muellerberndt/cadence/issues/14) | the integrated brain on the flagship platforms | one brain, three windows | open | |

### The rungs

| rung | feature | the demo | status | where |
| --- | --- | --- | --- | --- |
| 1 · [#7](https://github.com/muellerberndt/cadence/issues/7) | noticing itself: a governor patch reads the brain's own surprise and returns the mode | the dozing cat | accepted 2026-09-23; the fifth canonical example | `demos/dozing-cat/`, [live](https://floatingpragma.io/cadence-examples/dozing-cat/) |
| 2 · [#17](https://github.com/muellerberndt/cadence/issues/17) | weighing the senses: a steering patch sets each sense's gain inside the repair | the ventriloquist | accepted 2026-09-23 | `demos/ventriloquist/` |
| 3 · [#17](https://github.com/muellerberndt/cadence/issues/17) | looking: the steering patch moves where a sense samples, under a sensing cost | the lighthouse keeper | in progress | `demos/lighthouse-keeper/` |
| 4 · [#18](https://github.com/muellerberndt/cadence/issues/18) | the tiger: capture within one decision, habituation on repetition without consequence | the night nursery | in progress | `demos/night-nursery/` |
| 5 · [#22](https://github.com/muellerberndt/cadence/issues/22) | curiosity: the steering patch's own cost prefers falling, consequential surprise | the toy box | open | |
| 6 · [#11](https://github.com/muellerberndt/cadence/issues/11) | how long to think | the chess clock | open | |
| 7 · [#19](https://github.com/muellerberndt/cadence/issues/19) | knowing what it does not know | phone a friend | open | |
| 8 · [#20](https://github.com/muellerberndt/cadence/issues/20) | the self-model: the brain reports its own gaze, the grounded report as language | the sports commentator | open | |
| 9 · [#23](https://github.com/muellerberndt/cadence/issues/23) | the inner gaze: which futures to imagine | the heist | open | |
| 10 · [#24](https://github.com/muellerberndt/cadence/issues/24) | intention over intention | the DJ set | open | |
| 11 · [#21](https://github.com/muellerberndt/cadence/issues/21) | another observer | the magician | open | |
| 12 · [#21](https://github.com/muellerberndt/cadence/issues/21) | dialogue | twenty questions | open | |
| 13 · [#25](https://github.com/muellerberndt/cadence/issues/25) | teaching | the driving instructor | open | |
| 14 · [#26](https://github.com/muellerberndt/cadence/issues/26) | a body among things | fetch | open | |
| 15 · [#27](https://github.com/muellerberndt/cadence/issues/27) | the humanoid that acts and speaks among people | the kitchen helper | open | |
| 16 · [#28](https://github.com/muellerberndt/cadence/issues/28) | beyond: development, collectives, deep dialogue | the parliament | open | |

## The order of work

The rungs are climbed in order, and a rung's demo is built on the accepted demos below it: the lighthouse keeper on the ventriloquist's steering patch (the gain becomes a gaze), the night nursery on the cat's governor and the ventriloquist's gains, the toy box on the nursery. A substrate demo is built when a rung needs it: the polygraph as the pages multiply, ghost balls for the Atari form of rung 3, the alien zoo for rung 12, blindfold Connect Four and the chess clock together. Every accepted demo also asks what the library should carry, and the library releases follow those asks (0.14.0 carried the belief patch's gains, readback and admitted step from the first three).

## What each document is

- This page: the list of features, demos and status.
- [The ladder](METACOGNITION_LADDER.md): the science of each rung.
- The issues: one acceptance line per step, the discussion, the labels `ladder:*` and the milestones.
- The manifesto in the flagship folder: the mission and the design contract; the flagship's own plan file governs the paper and its evidence.
- The `demos/` folder: the code, the reports, the receipts, the pages.
