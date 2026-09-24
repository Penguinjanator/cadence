# The metacognition ladder

> **Final goal.** A patch-net equilibrium brain that is more scalable, more capable and more efficient than a transformer, climbing this ladder to a robot that acts and speaks among people. Every Cadence result is measured against that goal at matched information, matched task and a declared resource model.

The list of every feature with its demo and its status is [the plan](PLAN.md); this document orders the science, in three tracks. **The climb:** each rung is a function that the brain below it lacks, a task on which that lack is measured, an experiment to run, a control, a falsifier and a GitHub issue; the rungs run from brains that read the world only, where every measured Cadence result stands, to a humanoid that functions among people, and past it. **[The nursery](#the-nursery):** the order in which the world hands those functions to one brain in one life, with a caregiver in the loop. **[The breeder](#the-breeder):** what pays for a brain across lives. The ladder orders what a brain earns, the nursery what the world supplies, and the breeder what selection pays for. Nature builds a brain twice, across generations and within a life, and the program does both.

## The principle

**Metacognition is recursive self-observation.** A patch of the same kind as the others takes as its evidence the readback of the rest of the brain, the beliefs, repair residuals and surprises of the other patches, and its settled state returns to them as boundary conditions. We call it the steering patch. It is bounded, so it cannot hold the whole equilibrium below it at once. It must select the aspects that matter for what it is computing: which evidence counts, where a sense samples, what a habit holds, how long a repair runs. That selection is attention. Attention is a feature of the steering patch, never the whole of it.

The steering patch has a cost of its own. It prefers states in which the surprise below it is consequential and falling, so that sensing and computation go to what is learnable and what matters. That preference is curiosity.

Because the steering patch is a patch, its residual and surprise are readback in turn, and a patch can read the patch that steers. The levels form a ladder. Which readbacks exist is a question for the genome, never for the designer; the hand-designed steering patch is the control.

**Depth.** A patch that reads the world has depth 0. A patch whose evidence includes the readback of other patches has depth one more than the deepest patch it reads. A theorem in the flagship paper says what a level must do to be more than a reparameterization: a tower of *linear* readbacks that re-enter the drive additively collapses to one matrix, with the same equilibria and the same settling trajectories as a depth-0 patch. So a steering patch contributes nothing unless it publishes a lossy or nonlinear summary, acts on the equations below it (a gain that multiplies evidence, a gaze that changes which observation arrives, a goal that changes a clamped target, a budget that changes how long a repair runs), or carries a cost of its own.

**The unlock rule.** A rung is earned only by a task that the brain below it fails at matched information and compute and the brain with the rung passes. A shallow brain that solves a task placed higher falsifies that rung; a deeper brain that adds nothing on it at matched compute falsifies it too. Both outcomes are kept.

**The demo rule.** Every step's acceptance is its demo. Each page has the shared skeleton (the whole brain in the viewer, the world, the instrument strip, the plain-words account), a three-way switch (this step on; off, the brain below it; the hand-designed control) so the acceptance is visible on the page, and the mouse in the world. The step is accepted when the demo works with the switch on and fails with it off. The demos live under `oph-meta/demos/`. Each rung below names its demo; the substrate issues name theirs under "Before the climb".

**The prediction.** With the existence of each readback a gene, selection on the ladder's tasks is predicted to add a level exactly at the rung that needs it and nowhere below, and, with a developmental clock as a gene, to open the readbacks of deeper rungs late in a life, in the ladder's order. Children acquire these functions in the same order: alerting and orienting in infancy, executive attention from the end of the first year, error monitoring at twelve to eighteen months, asking for help at twenty months, mirror self-recognition at eighteen to twenty-four months, explicit false belief in the fourth and fifth year; prefrontal synapses peak after fifteen months and are pruned into adolescence, and association cortex matures after the sensory cortex it integrates. The flagship paper carries the sources. [The nursery](#the-nursery) states the order in which the world supplies these functions, from the self by contingency to words, and its order experiment tests whether that order is a mechanism.

## Where the existing brains stand

| Brain | Rung | Why |
| --- | --- | --- |
| The worm (the *C. elegans* connectome as one masked temporal patch net) | 0 | It reads its chemosensory ports and drives its muscles. No patch reads a patch. |
| Patch World (soft bodies with two record patches under selection) | 0 | Bodies read the world and act; selection acts across lives. No patch reads a patch. |
| The mouse (a mouse connectome in a labyrinth) | 0 | Place to turn. Its rung-7 experiment is below. |
| Connect Four (a value patch and a supplied search) | 0 | Inspected: the search reads the value patch where it stops looking and knows the rules; nothing reads the patch's residual, confidence or process. |
| The Amen composer (record patches, a sample instrument, an evolutionary critic) | 0 | Inspected: the critic scores rendered audio during selection across genomes and never the brain; within a life nothing reads a patch. |
| The cart-pole actor (a habit, a body model, a governor) | 1, hand-set | The governor reads the patch's own surprise and decides when to imagine and when to learn; its thresholds are set by hand. It is the control for rung 1. |
| The Atari player (a belief patch from pixels) | 0 | It repairs the whole frame at one resolution and imagines under candidate actions for a fixed horizon. With a moving window it is the platform of rung 3, with a thinking budget of rung 6, with keys and doors of rung 9. |
| The 1942 pilot (a records controller) | 0 | Frames to presses. |
| The language learner (records by day, weights by night) | 0 | It reads a corpus. Its rung-8 and rung-12 experiments are below. |
| The artist (an eye with a designed glance and pens) | 0 | The glance is designed; a learned gaze is rung 3. |
| A fly in the Matrix (the BANC connectome, 150,802 neurons, as one brain) | 0 | It flies a body and the mushroom body learns which smell means sugar on its own synapses. No patch reads a patch. |
| The dozing cat (a belief patch and a settling governor of fourteen neurons, every synapse a gene) | 1, accepted | The governor reads the patch's own surprise and returns the mode, selected against hand-set thresholds and a random search: catch rate 0.961 awake 0.238 of the time, against 0.962 always awake at 4.7 times the cost. |
| The ventriloquist (a steering patch over one cortex) | 2, accepted | The steering patch reads the eye's and the ear's surprise and sets each sense's gain inside the repair: error 0.041 against 0.081 with fixed gains. |

## Before the climb

Four issues are the substrate every rung stands on. They come first.

- [#7](https://github.com/muellerberndt/cadence/issues/7) One cortex that lives: the life loop that exposes the three signals every steering patch reads (repair residual, surprise, contrast validity) and carries the first rung, the governor.
- [#15](https://github.com/muellerberndt/cadence/issues/15) Instruments: one receipt schema and one page component that show those signals, plus the steering patch's own: gains sent down, gaze, sensing cost.
- [#9](https://github.com/muellerberndt/cadence/issues/9) Ports: the seam contract, extended so that a port may carry a patch's residual and surprise (the readback port). Needed from rung 2 whenever the steering patch reads more than one cortex.
- [#8](https://github.com/muellerberndt/cadence/issues/8) Habits as patches: the thing rungs 1 and 6 grant or withhold.

[#3](https://github.com/muellerberndt/cadence/issues/3) (the shared core) is carried by the belief patch of 0.13.0. [#10](https://github.com/muellerberndt/cadence/issues/10) (the diagnostic world) and [#12](https://github.com/muellerberndt/cadence/issues/12) (sleep) are rung-0 work that the climb uses and does not wait for.

**The substrate's demos.** [#7](https://github.com/muellerberndt/cadence/issues/7) The dozing cat: A cat on a sill, eyes half closed; your mouse is the laser dot. It wakes, chases, dozes. Works when the awake share is low and the catches high, and the always-awake control catches the same at three times the cost. The room with the heater and the window is the first rung-1 page and stays. [#15](https://github.com/muellerberndt/cadence/issues/15) The polygraph: The instrument strip on its own, clipped to any brain: you poke the world and the needles jump (residual, surprise, update, seam, gain, gaze, sensing cost). Works when the identical component sits on every other demo page. [#9](https://github.com/muellerberndt/cadence/issues/9) An eye and an ear watch one ball: Two record patches settled together through ports; you switch the lights or the sound off. Works when the eye keeps the ball in the dark through the port from the ear and loses it with the port cut, and when the muted ear hears a coming bounce through the port from the eye (still open). [#8](https://github.com/muellerberndt/cadence/issues/8) A reaching arm learns its muscle memory: The two-link arm reaches by planning, distils the planner into a habit, and you hang a weight on its hand. Works when the habit reaches at a fraction of the planner's cost, misses under the weight, and recovers after the body relearns and the habit is refitted, with the planner-only switch as the control. [#3](https://github.com/muellerberndt/cadence/issues/3) The shell game: Three cups and a ball; you shuffle the cups with the mouse and drop a screen over them. The belief patch's heat over the cups follows the shuffle, and while the screen is down its private imagination carries the ball. Works when it points to the right cup after hidden shuffles and the heat visibly moves with the cups. [#4](https://github.com/muellerberndt/cadence/issues/4) Ghost balls: Pong where the brain's imagined futures are drawn as fading ghost balls before it moves; a slider changes the physics mid-rally (gravity, a sticky wall). Works when the ghosts bend to the new physics within a few rallies and the paddle is there first. [#5](https://github.com/muellerberndt/cadence/issues/5) The bedtime story: You type five lines about who holds what, then ask who has the key now; the brain answers with a confidence bar. Lie in line three and the bar drops. Works when the calibration plot on the page is straight and a withheld combination is answered. [#6](https://github.com/muellerberndt/cadence/issues/6) Blindfold Connect Four: It was never told the rules; it learned the transitions. You play it while its imagined boards flicker beside the board and a cost meter runs against a search of the same strength. Works when it beats you at a fraction of the positions imagined. [#10](https://github.com/muellerberndt/cadence/issues/10) The alien zoo: Creatures are body times colour times behaviour, and the brain has never seen the blue hopping cube. You assemble one from parts and ask what it will do. Works when the withheld combinations are predicted and the two-property lookup control is not. [#12](https://github.com/muellerberndt/cadence/issues/12) Sleep on it: A sun-and-moon slider. The eye-and-ear brain sleeps on the page, the records visibly drain into the weights in the viewer, and in the morning it plays with its store wiped. Wake it early and see the deficit. Works when the morning brain matches the evening one. [#13](https://github.com/muellerberndt/cadence/issues/13) The breeder: Patch World as a family tree: you set the world (the price of mass, scarcity), watch the wiring change down the generations, and adopt a genome into a world it never saw. Works when the evolved wiring beats the hand-wired brain at matched compute, on screen. [#14](https://github.com/muellerberndt/cadence/issues/14) One brain, three windows: One integrated brain, three panes: Pong, the composer, the labyrinth. Cortices and ports are shared; you switch the window and the same regions light up. Works when each pane runs on the shared cortices and nothing is copied per task. [#16](https://github.com/muellerberndt/cadence/issues/16) The referee: The eye and the ear plus a third patch that decides whom to believe and where to look. You dim the lights, mute the room, move the ball. Works when the referee's gains track the truth and the fixed-gain brain gets fooled.

## The rungs

Each rung: what the steering patch reads and returns; what it unlocks; the experiments; the control and the falsifier; the issue.

### Rung 0. Reflex and habit

Reads the world only; returns predictions and actions. Tracking, habits, one-shot recall, grammar completion, control of a learned body. Every measured Cadence result stands here: the worm, Patch World, the mouse, Connect Four, the composer, the pilot, the language learner. Issues [#3](https://github.com/muellerberndt/cadence/issues/3), [#4](https://github.com/muellerberndt/cadence/issues/4), [#6](https://github.com/muellerberndt/cadence/issues/6), [#8](https://github.com/muellerberndt/cadence/issues/8), [#10](https://github.com/muellerberndt/cadence/issues/10), [#12](https://github.com/muellerberndt/cadence/issues/12).

**The demo.** the canonical examples and the three quickstart pages (the worm, Connect Four, the Amen composer, Patch World; `cadence-demo stream|decide|body`).


### Rung 1. Noticing itself

Reads its own surprise; returns its mode: habit, imagine, learn. Unlocks learning only when it is needed and imagination on demand; the routine life runs on the habit.

- **The dozing cat.** A brain runs on habit with half-closed eyes in a quiet room. A moving dot appears, a mouse. The surprise wakes it: it imagines, catches, and dozes again when the room is quiet. Measured: the share of decisions spent awake against the catches, and the wake latency.
- **The drummer.** A brain follows a steady beat by habit. The drummer changes tempo. Surprise triggers learning; measured: bars to catch up, and that no learning happens while the beat is steady.
- **The room with the heater and the window** ([#7](https://github.com/muellerberndt/cadence/issues/7)'s demo): the governor decides when to think.

Control: the hand-set governor of the cart-pole. Falsifier: the settling governor does no better than the hand-set thresholds at matched compute; the thresholds are then genes. Issue [#7](https://github.com/muellerberndt/cadence/issues/7).

**The demo.** The dozing cat. A cat on a sill, eyes half closed; your mouse is the laser dot. It wakes, chases, dozes. Works when the awake share is low and the catches high, and the always-awake control catches the same at three times the cost. The room with the heater and the window is the first rung-1 page and stays.


### Rung 2. Weighing the senses

Reads the residuals of several senses; returns the gain of each sense's evidence in repair. Unlocks trusting the right sense at the right time.

- **The ventriloquist.** An eye and an ear both report where a ball is. In some phases the ear is displaced, the dummy speaks. In the dark the eye is blind. The brain must weigh the eye when the two disagree in the light and the ear in the dark, and learn the switch it cannot see.
- **The two weather forecasters.** Two channels predict tomorrow; one lies on a hidden schedule.

Control: fixed gains; the designed inverse-variance rule; the steering patch shuffled across channels. Falsifier: fixed gains match held-out prediction at matched compute. Issue [#17](https://github.com/muellerberndt/cadence/issues/17), across cortices [#16](https://github.com/muellerberndt/cadence/issues/16).

**The demo.** The ventriloquist. you are the ventriloquist; drag the voice away from the dummy, dim the lights, and the gain sliders for eye and ear move on their own. Works when the brain follows the eye in the light and the ear in the dark while the fixed gains are fooled.


### Rung 3. Looking

Reads the residual map; returns where a sense samples, under a sensing cost. Unlocks a frame larger than the patch can repair and selective sensing.

- **The periscope.** Pong and Seaquest seen through a window the brain moves. Seaquest punishes a fixed gaze: oxygen, divers and enemies live in different corners of the frame.
- **The lighthouse keeper.** A dark sea; a beam the brain sweeps; ships approach from random bearings and must be lit before they reach the rocks. A smart sweep predicts where to look.
- **The Atari player with an eye.** The belief patch with a window; the frame size at which the whole-frame patch at matched compute falls below it is the unlock.
- **The pointing game** (the nursery's stage N3, [#35](https://github.com/muellerberndt/cadence/issues/35)). The same gaze follows the caregiver's gaze where it predicts something and drops it where it does not: the first cue about another observer, which children have at nine months, years before rung 11.

Control: the full frame at matched compute (a smaller patch), a random window, a center window, the steering patch cut. Falsifier: the full frame wins at every frame size tested. Issue [#17](https://github.com/muellerberndt/cadence/issues/17), the player [#4](https://github.com/muellerberndt/cadence/issues/4).

**The demo.** The lighthouse keeper. a dark sea, ships from random bearings, a beam the brain sweeps under a cost; you launch ships. Works when it lights them before the rocks and the random sweep does not.


### Rung 4. The tiger

A large surprise below seizes the steering patch's state within one decision; repetition without consequence habituates; a consequential event keeps capturing. Unlocks orienting and the interrupt.

- **The cocktail party.** The brain attends one conversation and ignores the other until its name, a rare cue that predicts reward, is spoken in the ignored one.
- **The smoke alarm and the ticking clock.** The clock ticks and is forgotten; the alarm predicts fire and never is.

Control: a hand-set threshold rule, its threshold a gene if it wins; the steering patch cut. Falsifier: the threshold rule matches capture and habituation at matched compute. Issue [#18](https://github.com/muellerberndt/cadence/issues/18).

**The demo.** The night nursery. A brain asleep in a house of sounds: a ticking clock, cars, a fridge; you play sounds by clicking. Works when the clock is forgotten within minutes and the baby's cry, which predicts consequence, wakes it every time.


### Rung 5. Curiosity

The steering patch's own cost prefers falling, consequential surprise; it sends the senses to what is learnable and leaves the mastered and the random. Unlocks learning what matters without a schedule.

- **The noisy television.** Two screens: static and a slow cartoon. A surprise seeker stares at the static forever. The steering patch is expected to switch to the cartoon, stay while it learns it, and leave when it has.
- **The toy box.** A rattle, a music box, a jack-in-the-box, a kaleidoscope and a mirror. The record is the order in which the brain leaves them: the kaleidoscope early, the mastered toys when mastered, the mirror last.

Control: a surprise maximizer, random sensing, the designed learning-progress rule. Falsifier: random sensing matches prediction gain per unit of sensing cost. Issue [#22](https://github.com/muellerberndt/cadence/issues/22).

**The demo.** The toy box. Five toys and a static television; the gaze wanders and you drop in a new toy. Works when it goes to the new toy, stays while it learns it, leaves when it has, and never gets stuck on the static as the surprise maximizer does.


### Rung 6. How long to think

Reads the residual and the stakes; returns repair rounds and imagination budget under a deadline, with the habit as the fallback. Unlocks cheap routine and deliberate hard cases.

- **The chess clock.** Connect Four with a time bank per game. The brain spends its thinking where the position is uncertain and the stakes high; quality against time is the curve.
- **The crossing.** A game in which hesitation kills; the deadline is the test of the fallback.

Control: fixed compute at the same average compute. Falsifier: fixed compute matches quality. Issue [#11](https://github.com/muellerberndt/cadence/issues/11).

**The demo.** The chess clock. Connect Four with a time bank, against you; a bar shows its thinking per move. Works when it thinks long where the position is uncertain and reaches the same quality at half the compute of fixed thinking.


### Rung 7. Knowing what it does not know

A patch reads the steering patch; returns a confidence, an opt-out, a request for help. Unlocks metacognitive sensitivity, measured as meta-d', and help-seeking.

- **The mouse that stops to sniff.** The connectome mouse in the labyrinth with an uncertain place belief. When it is lost it stops and looks around instead of running; measured: meta-d' between its confidence and its correct turns, and time to the cheese against a mouse that never stops.
- **Phone a friend.** A quiz with a costly lifeline; the oracle is asked where the confidence is low, and the asking falls as competence rises.
- **The bet.** A wager on its own decision after it is made.

Control: the confidence read straight from the depth-0 residual; a designed margin; random opt-out at the matched rate. Falsifier: the depth-0 confidence has equal meta-d'. Issue [#19](https://github.com/muellerberndt/cadence/issues/19).

**The demo.** Phone a friend. The labyrinth mouse stops to sniff when it is lost, and in a quiz you are the friend: when its confidence is low a bubble asks you. Works when its confidence tracks its correctness better than the plain residual does and the asking falls as it learns.


### Rung 8. The self-model

A patch reads the steering patch; returns a prediction of the brain's own next gaze and gain, and says it in words. Unlocks the attention schema and grounded language about the brain's own state. Grounded language about the world begins earlier, at the nursery's first words ([#36](https://github.com/muellerberndt/cadence/issues/36)), and the self this rung models is the second self: the ecological self by contingency is stage N0 of the nursery.

- **The sports commentator.** A brain narrates its own Pong game, "watching the ball", "looked at the paddle", graded against its logged gaze. A narrator that sees the frame and not the steering patch is the control: it can describe the world and cannot describe the gaze.
- **The eye tracker in the head.** The brain predicts its own gaze a decision ahead; the prediction fed back as a prior improves control of attention.
- **The composer's diary.** The composer says what it intends in the next bar, graded against what it does.
- **The mirror** (the nursery's stage N5, [#37](https://github.com/muellerberndt/cadence/issues/37)). The mark test with a live and a delayed mirror; the conceptual self, built on the ecological one.

Falsifier: the depth-0 narrator narrates the gaze as well. Issue [#20](https://github.com/muellerberndt/cadence/issues/20), the grammar [#5](https://github.com/muellerberndt/cadence/issues/5).

**The demo.** The sports commentator. Pong with captions the brain writes about its own gaze: watching the ball; lost it, looking at the paddle. Hide the ball. Works when the captions match the logged gaze and the frame-only narrator can describe the world but never where it looked.


### Rung 9. The inner gaze

Imagination is a sense the brain samples. The steering patch chooses which futures to imagine, how far, and when to stop, from the rung-7 confidence and the stakes. Unlocks strategy: a plan before the first move.

- **The heist.** A key, a locked door, a vault and a guard on a patrol loop. Being seen ends the life. The route and its timing are imagined before the first step and re-imagined when the guard deviates.
- **Sokoban for patches.** Boxes pushed into corners are lost for good; the plan must be complete before the push.
- **Connect Four by its own uncertainty.** With a learned transition, the lines imagined are chosen where the position is uncertain, against uniform search at the same number of imagined positions.

Control: uniform imagination at matched compute, random branch selection, the greedy patch, a designed tree search. Falsifier: uniform search is as good. Issue [#23](https://github.com/muellerberndt/cadence/issues/23), Connect Four [#6](https://github.com/muellerberndt/cadence/issues/6).

**The demo.** The heist. A top-down vault, a guard on patrol, a key. Ghost routes are imagined before the first step; you move the guard and it re-plans. Works when it gets in where uniform imagination at the same budget is caught.


### Rung 10. Intention over intention

A patch holds a goal for the goal-holder. Unlocks long-form creation, multi-stage tasks and the weighing of a goal's desirability.

- **The DJ set.** The composer holds a four-minute arc, intro, build, drop, breakdown, second drop, outro, above its bar-level intention; each bar serves the arc. Blind listeners rate the arcs against the bar-level composer with the same critic.
- **The mural.** The artist plans a composition, a focal point, thirds, a palette, and paints strokes that serve it.
- **The dungeon.** Rooms, keys and a boss; the room-level plan serves the dungeon-level intention.

Control: the bar-level composer; a designed arc template imposed from above; a shuffled arc-holder. Falsifier: the bar-level composer scores equal with listeners. Issue [#24](https://github.com/muellerberndt/cadence/issues/24), the Amen track [#14](https://github.com/muellerberndt/cadence/issues/14).

**The demo.** The DJ set. The composer plays a four-minute arc drawn as a mountain, each bar serving it; you ask for the drop now or reshape the mountain. Works when blind listeners prefer the arc brain to the bar-level composer.


### Rung 11. Another observer

A patch reads a model of another observer, a copy of the brain's own lower patches settled on what the partner could see, and returns a belief about the partner's belief. Unlocks false belief and another's attention. Its precursor, joint attention (following another's gaze, pointing, checking that the other looks), is the nursery's stage N3 ([#35](https://github.com/muellerberndt/cadence/issues/35)) and is measured first.

- **Hide and seek.** The hider models where the seeker will look; the seeker models where the hider thinks it will look.
- **The magician.** A brain that misdirects another brain's gaze to make a card disappear; it can do so only by modeling the other's attention.
- **Pong doubles.** Two brains, one ball, one paddle each; who takes the ball is a belief about the other.

Control: a predictor of the partner that uses the truth rather than the partner's evidence. Falsifier: the truth-based predictor is as good when the partner's gaze was elsewhere. Issue [#21](https://github.com/muellerberndt/cadence/issues/21).

**The demo.** The magician. your cursor is your gaze; the brain makes a card vanish by steering where you look, and plays hide and seek where you hide. Works when the trick fails against a brain that sees the truth instead of modelling your attention.


### Rung 12. Dialogue

Two brains exchange grounded sentences about one world and each updates its belief from the other's words. Unlocks world models by words. The words come from the nursery's first words and two words ([#36](https://github.com/muellerberndt/cadence/issues/36), [#38](https://github.com/muellerberndt/cadence/issues/38)) or from the grammar of #5.

- **The two lookouts.** An eye brain and an ear brain, each with a partial view of one ball, describe and ask in the grammar of rung 8; the measure is joint prediction against either alone and against two silent brains.
- **Battleship by radio.** Two brains with hidden boards; the words carry the game.
- **Twenty questions.** One brain holds a secret in the diagnostic world; the other asks.

Falsifier: no gain over silent brains at matched compute. Issue [#21](https://github.com/muellerberndt/cadence/issues/21), the grammar [#5](https://github.com/muellerberndt/cadence/issues/5).

**The demo.** Twenty questions. you hold a secret in the alien zoo and the brain asks in words; two brains play battleship by radio. Works when the words carry the game and two silent brains lose.


### Rung 13. Teaching

A patch reads the student's surprise and returns the next lesson and the words for it. Unlocks curricula and learning from teaching.

- **The driving instructor.** A competent cart-pole or Pong brain teaches a fresh one: which situation to put it in, when to demonstrate, when to let it fail, said in sentences; the student asks.
- **The piano teacher.** The composer teaches a novice a groove, a figure at a time, from the novice's surprise.

Control: a fixed curriculum, a random one, self-play, the teacher without the readback port, and the scripted caregiver of the nursery ([#32](https://github.com/muellerberndt/cadence/issues/32)), whose behaviours (mirroring, pointing, naming, the lesson chosen from the student's surprise) a teaching brain must produce by reading the student. Falsifier: the fixed curriculum is as fast. Issue [#25](https://github.com/muellerberndt/cadence/issues/25).

**The demo.** The driving instructor. A competent Pong brain teaches a fresh one, choosing each lesson from the student's surprise and saying it in sentences; you can take the student's seat. Works when the taught student learns faster than under a fixed curriculum.


### Rung 14. A body among things

Every lower rung on a real body. Unlocks the real world.

- **The blindfold course.** In simulation, a quadruped crosses obstacles it can see only through a gaze it moves; a sound predicts a moving obstacle; its confidence about the next footing decides whether it steps or first looks.
- **Fetch.** On the real dog: a person throws a ball; the dog attends to the ball and the person alternately, follows pointing, looks back or barks when it has lost the ball, and brings it.

Control: the imitation teacher alone; a designed state machine with the same sensors. Falsifier: the state machine matches. Issue [#26](https://github.com/muellerberndt/cadence/issues/26); the nursery's learned imitation ([#34](https://github.com/muellerberndt/cadence/issues/34)) and pointing ([#35](https://github.com/muellerberndt/cadence/issues/35)) serve it.

**The demo.** Fetch. The robot dog in the browser: you throw the ball with the mouse; it looks at the ball and at you by turns, barks when it has lost it, brings it back. Works when the state machine with the same sensors fails on the throws it never saw.


### Rung 15. The humanoid that acts and speaks among people

The end of the program. Hands, eyes and ears; language grounded in its own attention, plans and body; joint tasks with a person; it asks when unsure and says what it is doing. The nursery ([#30](https://github.com/muellerberndt/cadence/issues/30)) is how it is raised: its words are grounded there, and the order experiment says in which order.

- **The kitchen helper.** "Make me a coffee and tell me what you are doing."
- **Furniture with a friend.** Assemble a shelf with a person: hold the plan, ask for the next part, notice when the partner is stuck, explain the step.
- **The tour guide.** Lead a visitor through a room, reading where the visitor looks and what the visitor has understood.

Control: a scripted assistant with the same sensors and actuators; a transformer-based agent at a declared resource model. Falsifier: either matches on completion, truth of the sentences and asks-to-errors. Issue [#27](https://github.com/muellerberndt/cadence/issues/27).

**The demo.** The kitchen helper. Make me a coffee and tell me what you are doing, in simulation first, then on the robot. Works when the coffee is made, every sentence is true, and it asks exactly when it should.


### Rung 16 and beyond

A patch reads the effect of the genome across lives and makes development an object of metacognition; many brains read each other's steering patches; dialogue carries a depth of nested belief people do not sustain.

- **The brain that writes the next rung.** Given the instruments, a brain proposes the experiment on itself that most reduces its surprise about itself, and runs it.
- **The parliament.** A hundred brains share one costly sense and model each other's attention to allocate it, against an auction and a round robin.
- **Deep dialogue.** A hidden-information card game with several players, won or lost by the depth of nested belief at matched compute.

Issue [#28](https://github.com/muellerberndt/cadence/issues/28).

**The demo.** The parliament. A hundred brains share one telescope and model each other's attention to allocate it; you bribe one. Works when the allocation beats an auction and a round robin, and when the brain that proposes its own next experiment runs one you approve.

## The nursery

**The claim.** Development is a mechanism. The order in which the world hands functions to a brain, and the clock on which the brain's readbacks open, change what the brain can learn at matched interactions and compute. This is development, never pretraining: pretraining fits weights to a corpus the brain did not generate; development is one life in which the brain's own actions produce its data, a caregiver is in the loop, and stages open as the body matures and as earlier stages are mastered. It is a process that can be simulated at high speed, since nothing in it is tied to wall-clock time except the number of contingent interactions and the settling cost per decision. The developmental clock ticks in decisions and is a gene.

**Nature's order, with the sources.** Before birth, months of spontaneous movement with touch and proprioception grow body maps; a simulated fetus with a spinal cord and a body develops them from that alone (Kuniyoshi and Sangawa 2006, Biological Cybernetics; Yamada et al. 2016, Scientific Reports). From birth to five months, the self by contingency: what I do, I feel, in perfect temporal register. Newborns root more to another's touch than to their own hand (Rochat and Hespos 1997), two-month-olds learn that kicking moves a mobile (Rovee and Rovee 1969; Watson 1972), five-month-olds tell a live video of their own legs from a delayed one (Bahrick and Watson 1985, Developmental Psychology). This is the self-environment distinction, it comes first, and it is what makes imitation possible: there is nothing to map another's act onto without it. At two to three months, the contingent other: turn-taking and proto-conversation (Trevarthen 1979), distress when the caregiver's face goes still (Tronick et al. 1978), and the caregiver imitating the baby far more than the reverse (Pawlby 1977; Jones 2009). Imitation proper at nine to eighteen months: neonatal imitation is contested (Meltzoff and Moore 1977, Science, against Oostenbroek et al. 2016, Current Biology, who found none in 106 infants), robust imitation of novel acts and deferred imitation arrive around nine months (Meltzoff 1988), and the mirror mapping is learned from correlated seeing and doing, which being imitated supplies, and is reversed by counter-mirror training (Heyes 2010; Catmur, Walsh and Heyes 2007). Around nine months, joint attention: gaze following, pointing, and the tuning to ostensive cues that Csibra and Gergely (2009, Trends in Cognitive Sciences) call natural pedagogy (Tomasello 1999; Carpenter, Nagell and Tomasello 1998). At twelve months, first words, which are labels of jointly attended objects and never self-reports: eighteen-month-olds map a new word to what the speaker looks at, even when they themselves look elsewhere (Baldwin 1991, Child Development). At eighteen to twenty-four months, the conceptual self: the mirror mark test (Amsterdam 1972), which fails with a two-second video delay (Miyazaki and Hiraki 2006), then the vocabulary spurt and two-word speech (Brown 1973; Bates and Goodman 1997). Explicit false belief follows at four to five years (Wellman, Cross and Watson 2001).

**Two selves.** There are two selves, and the ladder's rung 8 models the second. The ecological self (Neisser 1988; Rochat's levels 0 and 1) is the part of the reading that efference explains, present at birth and calibrated by babbling; it is a rung-0 measurement that no existing brain has made, although the cat's belief patch explains almost all of the paw's change from its own push and Patch World's model reads this tick's motor. The conceptual self (the mirror, the name, the pronoun) comes after words and joint attention and is built on the first.

**The caregiver.** The missing object of the program. It is the environment, never a brain: it feeds, mirrors, responds contingently, looks and points, names what is jointly attended, and later asks and answers ([#32](https://github.com/muellerberndt/cadence/issues/32)). Scripting it breaks no principle, since the hand-written rule is on the world's side of the seam; from rung 13 on, a Cadence brain takes its place, and the scripted caregiver is that rung's control. Seen from a population, the caregiver is the provisioning gene of the breeder.

**The stages.** Each stage: what the world supplies, nature's age, the rung it uses, the measure, the control, the falsifier and the issue. A stage is built on the accepted demos below it and is accepted by its demo under the demo rule.

| stage | what the world supplies | age | uses | the measure | the control | the falsifier | issue |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N0 the ecological self | a body, babbling, touch, proprioception, a mobile on a ribbon | before birth to 5 months | rung 0 | the self split of the residual; surprise on a live against a delayed feed of itself; the push test | the patch without efference; the true body model as a given efference copy | the patch without efference detects contingency and the push as well | [#31](https://github.com/muellerberndt/cadence/issues/31) |
| the caregiver | the environment object | from birth | | declared behaviours, reproducible from a seed, with yoked controls built in | | | [#32](https://github.com/muellerberndt/cadence/issues/32) |
| N1 the contingent other | a face that answers, then goes still | 2 to 3 months | rungs 1 and 2 | the face predicted from efference; the still-face response; turn-taking | a yoked random face; a cross-correlation detector | the random face gives the same response | [#33](https://github.com/muellerberndt/cadence/issues/33) |
| N2 imitation learned | a caregiver that imitates the baby | 9 to 18 months | rung 0 records | the mirrored baby imitates the repertoire, a novel act, a deferred act; the counter-mirrored baby imitates crossed | the never-mirrored baby; the innate mirror through the true kinematics | the never-mirrored baby imitates as well | [#34](https://github.com/muellerberndt/cadence/issues/34) |
| N3 joint attention | a gaze that predicts events, a hand that fetches | 9 to 12 months | rung 3 | gaze following above the cost-optimal sweep; pointing that gets things; checking | an uninformative gaze; a fixed gaze; always-follow | the fixed gaze does as well, or following persists when the gaze is uninformative | [#35](https://github.com/muellerberndt/cadence/issues/35) |
| N4 first words | a name said under joint attention | 12 to 18 months | rung 0 records and rung 3 | comprehension, production, Baldwin's discrepant-labeling test, fast mapping | naming without joint attention; the baby without N3; a co-occurrence table | the co-occurrence table passes Baldwin's test | [#36](https://github.com/muellerberndt/cadence/issues/36) |
| N5 the conceptual self | a mirror | 18 to 24 months | rung 8 | the mirror's contingency detected; the mark test, live against delayed | the brain without efference; a hand-written matcher | the delayed mirror passes the mark test | [#37](https://github.com/muellerberndt/cadence/issues/37) |
| N6 grammar | two-word utterances and a night | about 2 years | rung 0, the reading brain | withheld combinations understood and produced; morning against evening | no night; a co-occurrence table over pairs | the evening brain produces them as well | [#38](https://github.com/muellerberndt/cadence/issues/38) |
| N7 the order experiment | the same interactions in nature's order, shuffled, or all at once | | the clock | the battery of N0 to N6 at matched counts and compute | the shuffled childhood; the born-open brain; no womb | the born-open brain matches nature's order | [#39](https://github.com/muellerberndt/cadence/issues/39) |

**The order experiment.** Four arms at matched interaction counts and compute: nature's order with the readbacks opening on the clock; the shuffled childhood; the born-open brain with every readback and every stage available from the first moment (the control [#13](https://github.com/muellerberndt/cadence/issues/13) names); and nature's order without the womb. The prediction, stated before the run: nature's order wins on the late stages (words, the mirror, two words) at matched counts, and the arm without the womb falls below on the self split and the mirror. If the born-open brain matches on the whole battery, development is decoration in this substrate, the clock is dropped from the genome, and the nursery is recorded as a curriculum rather than a mechanism. Either outcome is a result. With the opening moments of the readbacks as genes, the breeder's machinery runs the clock's evolution against the hand-set clock.

**Prior art.** This track has a name, developmental robotics: Lungarella, Metta, Pfeifer and Sandini (2003, Connection Science), Asada et al. (2009, IEEE Transactions on Autonomous Mental Development), Cangelosi and Schlesinger (2015, MIT Press), Oudeyer, Kaplan and Hafner (2007, IEEE Transactions on Evolutionary Computation) on intrinsic motivation, and the iCub. Twenty years of it, with designed architectures and no matched controls. What Cadence brings is one rule, an evolvable genome and clock, and a falsifier per stage.

**The demos.** [#31](https://github.com/muellerberndt/cadence/issues/31) The mobile: a baby in a crib, its hand tied to a mobile; your mouse pushes the hand or shakes the mobile; the instrument strip shows the surprise splitting into "me" and "not me". [#33](https://github.com/muellerberndt/cadence/issues/33) The still face: a face above the crib mirrors the baby; you freeze it. [#34](https://github.com/muellerberndt/cadence/issues/34) The mirrored baby: you are the caregiver's arm. [#35](https://github.com/muellerberndt/cadence/issues/35) The pointing game: you are the face and the hand. [#36](https://github.com/muellerberndt/cadence/issues/36) The naming game: you name toys while looking at them. [#37](https://github.com/muellerberndt/cadence/issues/37) The mirror: you put a sticker on the baby's forehead. [#38](https://github.com/muellerberndt/cadence/issues/38) Two words: "ball red", then the sun-and-moon slider. [#39](https://github.com/muellerberndt/cadence/issues/39) The shuffled childhood: three babies raised in three orders, tested with the same games. [#30](https://github.com/muellerberndt/cadence/issues/30) The nursery: one page, one baby, one long life, and you are the caregiver.

## The breeder

**What the receipts say.** Three evolving worlds ran the selection experiment, and selection removed brains every time. In cadence-world, brains shrank at the page's prices, biting was the only rule that grew them, two-cortex brains stayed at mutation supply, and symbols carried no meaning. In cadence-worms, no lever in 21 probe batches selected a cortex, predation included; carriers died at half the age of reflex worms, and the recorded verdict is that a learner worse than the reflex at birth never earns its price. In Patch World, records lower the model's late-life error and carriers die younger in every world; over 20,000 ticks brains shrank and the records gene went to zero in most lineages. The genome has the cheap mutation, as nature did; the niche is what is missing. A physics-and-food world has a fixed complexity, and once a reflex meets it, a brain never pays again. In cadence-world's fifth sweep, cheaper reading let biting grow multi-cortex brains that lived as long as small ones: the one rule that pits a brain against a brain is the one that grew brains.

**What grew brains.** Two pressures in the record broke the ceiling. Other minds: the only part of an environment whose difficulty rises with your own competence is another learning brain; within-life learning pays only where the world varies within a life, and agents that learn are the unbounded source of that variation (Humphrey 1976; Dunbar 1998; Muthukrishna, Doebeli, Chudek and Henrich 2018; Van Valen 1973). A subsidised childhood: a brain that loses to a reflex on day one is affordable only if someone else pays for day one (Kaplan, Hill, Lancaster and Hurtado 2000). Culture rides on both. The mutation that added cortex was a duplication of fields (Krubitzer 2007) driven by a handful of genes that add progenitors and lengthen neurogenesis (Florio et al. 2015; Fiddes et al. 2018; Suzuki et al. 2018), and the added tissue, far from every sensory port, became cortex that reads cortex: the tethering hypothesis (Buckner and Krienen 2013), the ladder's depth as an evolutionary story. In the adult the same order is the principal gradient from sensory cortex to the default network (Margulies et al. 2016).

**The experiments.** Under [#13](https://github.com/muellerberndt/cadence/issues/13); the fitness stays energy and death, and every outcome is kept.

| experiment | the step | the control | the falsifier | issue |
| --- | --- | --- | --- | --- |
| B1 the Red Queen | two co-evolving populations in Patch World, predators and prey, then contested food, then shared kills; escalation measured against a saved panel of past champions so that cycling is told apart (Rosin and Belew 1997; Cliff and Miller 1995) | the opposing population's genome frozen; scripted predators | brains grow as much against frozen opponents, or not at all | [#40](https://github.com/muellerberndt/cadence/issues/40) |
| B2 the duplication mutation | one operator: copy a cortex, and with probability p rewire its inputs to other cortices' states, residuals and surprises and its outputs to their gains and budgets, multiplicatively, so the linear-tower theorem does not collapse it | the collapsed operator with additive outputs; the plain world; random search | readback copies never rise above mutation supply in any world, and depth must be designed | [#41](https://github.com/muellerberndt/cadence/issues/41) |
| B3 provisioning as a gene | a gene T: a parent feeds its offspring for T ticks and the juvenile reads the parent's pose and motor | T fixed at zero; T fixed high; the hand-set T | T never lengthens, or the records gene stays at zero at every T | [#42](https://github.com/muellerberndt/cadence/issues/42) |
| B4 the recapitulation check | per cortex, from the chronicles: the generation added, the graph distance from the ports, the depth, the opening moment; the prediction that late-added cortices sit far from the ports and read cortex, the human gradient as the control and never the blueprint | a random-addition null; the plain world | no correlation in any world | [#43](https://github.com/muellerberndt/cadence/issues/43) |

**What the breeder is not.** Not a humanoid body first: the fly, the worm and the dog show that the body side works and was never the bottleneck; a humanoid among scripted bots is again a fixed-complexity world and would reproduce the negative result at a higher cost per life. Evolve in the cheap world, then develop the evolved genome into the expensive body, which is what rung 14 says. Not a rebuilt human brain: below the macro level it is not available, the one human connectome is a cubic millimetre (Shapson-Coe et al. 2024), and the macro gradient is the control.

**The demos.** [#40](https://github.com/muellerberndt/cadence/issues/40) The arms race: predators and prey on the torus; you freeze the predators' genome with a click. [#41](https://github.com/muellerberndt/cadence/issues/41) The copy: click a being and see a new cortex that reads the others; the family tree shows when readback copies were kept. [#42](https://github.com/muellerberndt/cadence/issues/42) The long childhood: a slider forces the childhood long or short. [#43](https://github.com/muellerberndt/cadence/issues/43) The gradient: the family tree coloured by generation of addition beside the brain coloured by distance from the ports. [#13](https://github.com/muellerberndt/cadence/issues/13) The breeder: Patch World as a family tree.

## Cross-cutting issues

- [#16](https://github.com/muellerberndt/cadence/issues/16) The steering patch across cortices: rungs 2 to 5 on the joint equilibrium of several cortices, with [#17](https://github.com/muellerberndt/cadence/issues/17) as the one-cortex proof.
- [#13](https://github.com/muellerberndt/cadence/issues/13) Evolution: which readbacks exist and when they open are genes; the ladder's prediction is tested here, with the hand-designed steering patch and the born-open brain as controls.
- [#15](https://github.com/muellerberndt/cadence/issues/15) Instruments: the signals of every rung on every page and in every receipt.
- [#9](https://github.com/muellerberndt/cadence/issues/9) Ports: the readback port.
- [#30](https://github.com/muellerberndt/cadence/issues/30) The nursery: the umbrella of stages N0 to N7 and the caregiver [#32](https://github.com/muellerberndt/cadence/issues/32).
- [#40](https://github.com/muellerberndt/cadence/issues/40) to [#43](https://github.com/muellerberndt/cadence/issues/43) The breeder's four experiments under [#13](https://github.com/muellerberndt/cadence/issues/13).
- [#2](https://github.com/muellerberndt/cadence/issues/2) The parent issue carries the order, the three tracks and the dependency graph; every dependency is recorded on GitHub as a "blocked by" relation and as a sub-issue.

## Reading the ladder

Labels on the issues give the stage: `ladder:substrate` (rung 0), `ladder:selection` (rungs 1 to 5), `ladder:self-knowledge` (rungs 6 to 9), `ladder:intention` (rung 10), `ladder:others` (rungs 11 to 13), `ladder:world` (rungs 14 and 15), `ladder:beyond` (rung 16); `nursery` and `breeder` mark the two other tracks, and milestone M10 collects the nursery. Milestones name the engineering deliverables; the ladder names the order of the science. Every rung keeps the principles: one patch rule, settled states as answers, evolution before design, genes for anything that looks designed, and every outcome kept. A new rung is added here when a task is found that the brain of the rung above fails at matched information and compute; a rung is removed here when its task is solved by the brain below it.


