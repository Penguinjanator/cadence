# The metacognition ladder

> **Final goal.** A patch-net equilibrium brain that is more scalable, more capable and more efficient than a transformer, climbing this ladder to a robot that acts and speaks among people. Every Cadence result is measured against that goal at matched information, matched task and a declared resource model.

This document orders the work of the Cadence program. Each rung is a function that the brain below it lacks, a task on which that lack is measured, an experiment to run, a control, a falsifier and a GitHub issue. The rungs run from brains that read the world only, where every measured Cadence result stands, to a humanoid that functions among people, and past it.

## The principle

**Metacognition is recursive self-observation.** A patch of the same kind as the others takes as its evidence the readback of the rest of the brain, the beliefs, repair residuals and surprises of the other patches, and its settled state returns to them as boundary conditions. We call it the steering patch. It is bounded, so it cannot hold the whole equilibrium below it at once. It must select the aspects that matter for what it is computing: which evidence counts, where a sense samples, what a habit holds, how long a repair runs. That selection is attention. Attention is a feature of the steering patch, never the whole of it.

The steering patch has a cost of its own. It prefers states in which the surprise below it is consequential and falling, so that sensing and computation go to what is learnable and what matters. That preference is curiosity.

Because the steering patch is a patch, its residual and surprise are readback in turn, and a patch can read the patch that steers. The levels form a ladder. Which readbacks exist is a question for the genome, never for the designer; the hand-designed steering patch is the control.

**Depth.** A patch that reads the world has depth 0. A patch whose evidence includes the readback of other patches has depth one more than the deepest patch it reads. A theorem in the flagship paper says what a level must do to be more than a reparameterization: a tower of *linear* readbacks that re-enter the drive additively collapses to one matrix, with the same equilibria and the same settling trajectories as a depth-0 patch. So a steering patch contributes nothing unless it publishes a lossy or nonlinear summary, acts on the equations below it (a gain that multiplies evidence, a gaze that changes which observation arrives, a goal that changes a clamped target, a budget that changes how long a repair runs), or carries a cost of its own.

**The unlock rule.** A rung is earned only by a task that the brain below it fails at matched information and compute and the brain with the rung passes. A shallow brain that solves a task placed higher falsifies that rung; a deeper brain that adds nothing on it at matched compute falsifies it too. Both outcomes are kept.

**The prediction.** With the existence of each readback a gene, selection on the ladder's tasks is predicted to add a level exactly at the rung that needs it and nowhere below, and, with a developmental clock as a gene, to open the readbacks of deeper rungs late in a life, in the ladder's order. Children acquire these functions in the same order: alerting and orienting in infancy, executive attention from the end of the first year, error monitoring at twelve to eighteen months, asking for help at twenty months, mirror self-recognition at eighteen to twenty-four months, explicit false belief in the fourth and fifth year; prefrontal synapses peak after fifteen months and are pruned into adolescence, and association cortex matures after the sensory cortex it integrates. The flagship paper carries the sources.

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

## Before the climb

Four issues are the substrate every rung stands on. They come first.

- [#7](https://github.com/muellerberndt/cadence/issues/7) One cortex that lives: the life loop that exposes the three signals every steering patch reads (repair residual, surprise, contrast validity) and carries the first rung, the governor.
- [#15](https://github.com/muellerberndt/cadence/issues/15) Instruments: one receipt schema and one page component that show those signals, plus the steering patch's own: gains sent down, gaze, sensing cost.
- [#9](https://github.com/muellerberndt/cadence/issues/9) Ports: the seam contract, extended so that a port may carry a patch's residual and surprise (the readback port). Needed from rung 2 whenever the steering patch reads more than one cortex.
- [#8](https://github.com/muellerberndt/cadence/issues/8) Habits as patches: the thing rungs 1 and 6 grant or withhold.

[#3](https://github.com/muellerberndt/cadence/issues/3) (the shared core) is carried by the belief patch of 0.13.0. [#10](https://github.com/muellerberndt/cadence/issues/10) (the diagnostic world) and [#12](https://github.com/muellerberndt/cadence/issues/12) (sleep) are rung-0 work that the climb uses and does not wait for.

## The rungs

Each rung: what the steering patch reads and returns; what it unlocks; the experiments; the control and the falsifier; the issue.

### Rung 0. Reflex and habit

Reads the world only; returns predictions and actions. Tracking, habits, one-shot recall, grammar completion, control of a learned body. Every measured Cadence result stands here: the worm, Patch World, the mouse, Connect Four, the composer, the pilot, the language learner. Issues [#3](https://github.com/muellerberndt/cadence/issues/3), [#4](https://github.com/muellerberndt/cadence/issues/4), [#6](https://github.com/muellerberndt/cadence/issues/6), [#8](https://github.com/muellerberndt/cadence/issues/8), [#10](https://github.com/muellerberndt/cadence/issues/10), [#12](https://github.com/muellerberndt/cadence/issues/12).

### Rung 1. Noticing itself

Reads its own surprise; returns its mode: habit, imagine, learn. Unlocks learning only when it is needed and imagination on demand; the routine life runs on the habit.

- **The dozing cat.** A brain runs on habit with half-closed eyes in a quiet room. A moving dot appears, a mouse. The surprise wakes it: it imagines, catches, and dozes again when the room is quiet. Measured: the share of decisions spent awake against the catches, and the wake latency.
- **The drummer.** A brain follows a steady beat by habit. The drummer changes tempo. Surprise triggers learning; measured: bars to catch up, and that no learning happens while the beat is steady.
- **The room with the heater and the window** ([#7](https://github.com/muellerberndt/cadence/issues/7)'s demo): the governor decides when to think.

Control: the hand-set governor of the cart-pole. Falsifier: the settling governor does no better than the hand-set thresholds at matched compute; the thresholds are then genes. Issue [#7](https://github.com/muellerberndt/cadence/issues/7).

### Rung 2. Weighing the senses

Reads the residuals of several senses; returns the gain of each sense's evidence in repair. Unlocks trusting the right sense at the right time.

- **The ventriloquist.** An eye and an ear both report where a ball is. In some phases the ear is displaced, the dummy speaks. In the dark the eye is blind. The brain must weigh the eye when the two disagree in the light and the ear in the dark, and learn the switch it cannot see.
- **The two weather forecasters.** Two channels predict tomorrow; one lies on a hidden schedule.

Control: fixed gains; the designed inverse-variance rule; the steering patch shuffled across channels. Falsifier: fixed gains match held-out prediction at matched compute. Issue [#17](https://github.com/muellerberndt/cadence/issues/17), across cortices [#16](https://github.com/muellerberndt/cadence/issues/16).

### Rung 3. Looking

Reads the residual map; returns where a sense samples, under a sensing cost. Unlocks a frame larger than the patch can repair and selective sensing.

- **The periscope.** Pong and Seaquest seen through a window the brain moves. Seaquest punishes a fixed gaze: oxygen, divers and enemies live in different corners of the frame.
- **The lighthouse keeper.** A dark sea; a beam the brain sweeps; ships approach from random bearings and must be lit before they reach the rocks. A smart sweep predicts where to look.
- **The Atari player with an eye.** The belief patch with a window; the frame size at which the whole-frame patch at matched compute falls below it is the unlock.

Control: the full frame at matched compute (a smaller patch), a random window, a center window, the steering patch cut. Falsifier: the full frame wins at every frame size tested. Issue [#17](https://github.com/muellerberndt/cadence/issues/17), the player [#4](https://github.com/muellerberndt/cadence/issues/4).

### Rung 4. The tiger

A large surprise below seizes the steering patch's state within one decision; repetition without consequence habituates; a consequential event keeps capturing. Unlocks orienting and the interrupt.

- **The cocktail party.** The brain attends one conversation and ignores the other until its name, a rare cue that predicts reward, is spoken in the ignored one.
- **The smoke alarm and the ticking clock.** The clock ticks and is forgotten; the alarm predicts fire and never is.

Control: a hand-set threshold rule, its threshold a gene if it wins; the steering patch cut. Falsifier: the threshold rule matches capture and habituation at matched compute. Issue [#18](https://github.com/muellerberndt/cadence/issues/18).

### Rung 5. Curiosity

The steering patch's own cost prefers falling, consequential surprise; it sends the senses to what is learnable and leaves the mastered and the random. Unlocks learning what matters without a schedule.

- **The noisy television.** Two screens: static and a slow cartoon. A surprise seeker stares at the static forever. The steering patch is expected to switch to the cartoon, stay while it learns it, and leave when it has.
- **The toy box.** A rattle, a music box, a jack-in-the-box, a kaleidoscope and a mirror. The record is the order in which the brain leaves them: the kaleidoscope early, the mastered toys when mastered, the mirror last.

Control: a surprise maximizer, random sensing, the designed learning-progress rule. Falsifier: random sensing matches prediction gain per unit of sensing cost. Issue [#22](https://github.com/muellerberndt/cadence/issues/22).

### Rung 6. How long to think

Reads the residual and the stakes; returns repair rounds and imagination budget under a deadline, with the habit as the fallback. Unlocks cheap routine and deliberate hard cases.

- **The chess clock.** Connect Four with a time bank per game. The brain spends its thinking where the position is uncertain and the stakes high; quality against time is the curve.
- **The crossing.** A game in which hesitation kills; the deadline is the test of the fallback.

Control: fixed compute at the same average compute. Falsifier: fixed compute matches quality. Issue [#11](https://github.com/muellerberndt/cadence/issues/11).

### Rung 7. Knowing what it does not know

A patch reads the steering patch; returns a confidence, an opt-out, a request for help. Unlocks metacognitive sensitivity, measured as meta-d', and help-seeking.

- **The mouse that stops to sniff.** The connectome mouse in the labyrinth with an uncertain place belief. When it is lost it stops and looks around instead of running; measured: meta-d' between its confidence and its correct turns, and time to the cheese against a mouse that never stops.
- **Phone a friend.** A quiz with a costly lifeline; the oracle is asked where the confidence is low, and the asking falls as competence rises.
- **The bet.** A wager on its own decision after it is made.

Control: the confidence read straight from the depth-0 residual; a designed margin; random opt-out at the matched rate. Falsifier: the depth-0 confidence has equal meta-d'. Issue [#19](https://github.com/muellerberndt/cadence/issues/19).

### Rung 8. The self-model

A patch reads the steering patch; returns a prediction of the brain's own next gaze and gain, and says it in words. Unlocks the attention schema and grounded language.

- **The sports commentator.** A brain narrates its own Pong game, "watching the ball", "looked at the paddle", graded against its logged gaze. A narrator that sees the frame and not the steering patch is the control: it can describe the world and cannot describe the gaze.
- **The eye tracker in the head.** The brain predicts its own gaze a decision ahead; the prediction fed back as a prior improves control of attention.
- **The composer's diary.** The composer says what it intends in the next bar, graded against what it does.

Falsifier: the depth-0 narrator narrates the gaze as well. Issue [#20](https://github.com/muellerberndt/cadence/issues/20), the grammar [#5](https://github.com/muellerberndt/cadence/issues/5).

### Rung 9. The inner gaze

Imagination is a sense the brain samples. The steering patch chooses which futures to imagine, how far, and when to stop, from the rung-7 confidence and the stakes. Unlocks strategy: a plan before the first move.

- **The heist.** A key, a locked door, a vault and a guard on a patrol loop. Being seen ends the life. The route and its timing are imagined before the first step and re-imagined when the guard deviates.
- **Sokoban for patches.** Boxes pushed into corners are lost for good; the plan must be complete before the push.
- **Connect Four by its own uncertainty.** With a learned transition, the lines imagined are chosen where the position is uncertain, against uniform search at the same number of imagined positions.

Control: uniform imagination at matched compute, random branch selection, the greedy patch, a designed tree search. Falsifier: uniform search is as good. Issue [#23](https://github.com/muellerberndt/cadence/issues/23), Connect Four [#6](https://github.com/muellerberndt/cadence/issues/6).

### Rung 10. Intention over intention

A patch holds a goal for the goal-holder. Unlocks long-form creation, multi-stage tasks and the weighing of a goal's desirability.

- **The DJ set.** The composer holds a four-minute arc, intro, build, drop, breakdown, second drop, outro, above its bar-level intention; each bar serves the arc. Blind listeners rate the arcs against the bar-level composer with the same critic.
- **The mural.** The artist plans a composition, a focal point, thirds, a palette, and paints strokes that serve it.
- **The dungeon.** Rooms, keys and a boss; the room-level plan serves the dungeon-level intention.

Control: the bar-level composer; a designed arc template imposed from above; a shuffled arc-holder. Falsifier: the bar-level composer scores equal with listeners. Issue [#24](https://github.com/muellerberndt/cadence/issues/24), the Amen track [#14](https://github.com/muellerberndt/cadence/issues/14).

### Rung 11. Another observer

A patch reads a model of another observer, a copy of the brain's own lower patches settled on what the partner could see, and returns a belief about the partner's belief. Unlocks false belief and another's attention.

- **Hide and seek.** The hider models where the seeker will look; the seeker models where the hider thinks it will look.
- **The magician.** A brain that misdirects another brain's gaze to make a card disappear; it can do so only by modeling the other's attention.
- **Pong doubles.** Two brains, one ball, one paddle each; who takes the ball is a belief about the other.

Control: a predictor of the partner that uses the truth rather than the partner's evidence. Falsifier: the truth-based predictor is as good when the partner's gaze was elsewhere. Issue [#21](https://github.com/muellerberndt/cadence/issues/21).

### Rung 12. Dialogue

Two brains exchange grounded sentences about one world and each updates its belief from the other's words. Unlocks world models by words.

- **The two lookouts.** An eye brain and an ear brain, each with a partial view of one ball, describe and ask in the grammar of rung 8; the measure is joint prediction against either alone and against two silent brains.
- **Battleship by radio.** Two brains with hidden boards; the words carry the game.
- **Twenty questions.** One brain holds a secret in the diagnostic world; the other asks.

Falsifier: no gain over silent brains at matched compute. Issue [#21](https://github.com/muellerberndt/cadence/issues/21), the grammar [#5](https://github.com/muellerberndt/cadence/issues/5).

### Rung 13. Teaching

A patch reads the student's surprise and returns the next lesson and the words for it. Unlocks curricula and learning from teaching.

- **The driving instructor.** A competent cart-pole or Pong brain teaches a fresh one: which situation to put it in, when to demonstrate, when to let it fail, said in sentences; the student asks.
- **The piano teacher.** The composer teaches a novice a groove, a figure at a time, from the novice's surprise.

Control: a fixed curriculum, a random one, self-play, the teacher without the readback port. Falsifier: the fixed curriculum is as fast. Issue [#25](https://github.com/muellerberndt/cadence/issues/25).

### Rung 14. A body among things

Every lower rung on a real body. Unlocks the real world.

- **The blindfold course.** In simulation, a quadruped crosses obstacles it can see only through a gaze it moves; a sound predicts a moving obstacle; its confidence about the next footing decides whether it steps or first looks.
- **Fetch.** On the real dog: a person throws a ball; the dog attends to the ball and the person alternately, follows pointing, looks back or barks when it has lost the ball, and brings it.

Control: the imitation teacher alone; a designed state machine with the same sensors. Falsifier: the state machine matches. Issue [#26](https://github.com/muellerberndt/cadence/issues/26).

### Rung 15. The humanoid that acts and speaks among people

The end of the program. Hands, eyes and ears; language grounded in its own attention, plans and body; joint tasks with a person; it asks when unsure and says what it is doing.

- **The kitchen helper.** "Make me a coffee and tell me what you are doing."
- **Furniture with a friend.** Assemble a shelf with a person: hold the plan, ask for the next part, notice when the partner is stuck, explain the step.
- **The tour guide.** Lead a visitor through a room, reading where the visitor looks and what the visitor has understood.

Control: a scripted assistant with the same sensors and actuators; a transformer-based agent at a declared resource model. Falsifier: either matches on completion, truth of the sentences and asks-to-errors. Issue [#27](https://github.com/muellerberndt/cadence/issues/27).

### Rung 16 and beyond

A patch reads the effect of the genome across lives and makes development an object of metacognition; many brains read each other's steering patches; dialogue carries a depth of nested belief people do not sustain.

- **The brain that writes the next rung.** Given the instruments, a brain proposes the experiment on itself that most reduces its surprise about itself, and runs it.
- **The parliament.** A hundred brains share one costly sense and model each other's attention to allocate it, against an auction and a round robin.
- **Deep dialogue.** A hidden-information card game with several players, won or lost by the depth of nested belief at matched compute.

Issue [#28](https://github.com/muellerberndt/cadence/issues/28).

## Cross-cutting issues

- [#16](https://github.com/muellerberndt/cadence/issues/16) The steering patch across cortices: rungs 2 to 5 on the joint equilibrium of several cortices, with [#17](https://github.com/muellerberndt/cadence/issues/17) as the one-cortex proof.
- [#13](https://github.com/muellerberndt/cadence/issues/13) Evolution: which readbacks exist and when they open are genes; the ladder's prediction is tested here, with the hand-designed steering patch and the born-open brain as controls.
- [#15](https://github.com/muellerberndt/cadence/issues/15) Instruments: the signals of every rung on every page and in every receipt.
- [#9](https://github.com/muellerberndt/cadence/issues/9) Ports: the readback port.
- [#2](https://github.com/muellerberndt/cadence/issues/2) The parent issue carries the order.

## Reading the ladder

Labels on the issues give the stage: `ladder:substrate` (rung 0), `ladder:selection` (rungs 1 to 5), `ladder:self-knowledge` (rungs 6 to 9), `ladder:intention` (rung 10), `ladder:others` (rungs 11 to 13), `ladder:world` (rungs 14 and 15), `ladder:beyond` (rung 16). Milestones name the engineering deliverables; the ladder names the order of the science. Every rung keeps the principles: one patch rule, settled states as answers, evolution before design, genes for anything that looks designed, and every outcome kept. A new rung is added here when a task is found that the brain of the rung above fails at matched information and compute; a rung is removed here when its task is solved by the brain below it.
