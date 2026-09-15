"""Learn a world, ground a word, remember one sighting and act on a request.

The room IDs, word/object tokens, goal protocol and search algorithm are supplied.
Only observed transitions, ambiguous scenes and sightings teach the synaptic stores.
This is a small integration example, not learned perception or fluent language.
"""

from copy import deepcopy

import numpy as np

import cadence as cd
from cadence.circuits import imagine


def cue(index: int, width: int) -> np.ndarray:
    return np.eye(width)[[index]]


def route(model: cd.SynapticMemory, start: int, goal: int) -> tuple[int, ...] | None:
    """Search reads acquired consequences; it never calls or updates the real world."""
    def predicted(room, action):
        return model.recall(cue(2 * room + action, 6))[0]

    plan = imagine(
        (start, 0),
        lambda s: [a for a in range(2) if predicted(s[0], a).max() > 0.5],
        lambda s, a: (int(predicted(s[0], a).argmax()), s[1] + 1),
        lambda s: float(s[0] == goal) - 0.01 * s[1],
        lambda s: s[0] == goal,
        depth=3,
    )
    if start == goal:
        return ()
    return next((f.sequence for f in plan.futures if f.state[0] == goal), None)


def request(model, words, events, room):
    meanings = words.recall(cue(0, 1))[0]  # the heard word ID
    winners = np.flatnonzero(np.isclose(meanings, meanings.max()))
    if meanings.max() <= 0 or len(winners) != 1:
        return None  # insufficient grounding: do not invent a referent
    location = events.recall(cue(int(winners[0]), 3))[0]
    return route(model, room, int(location.argmax())) if location.max() > 0.5 else None


def run(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    model = cd.SynapticMemory(np.arange(6), np.arange(6, 9), decay=1)
    words = cd.SynapticMemory(np.arange(1), np.arange(1, 4), decay=1, rate=0.25)
    events = cd.FastSynapses(np.arange(3), np.arange(3, 6), rule="delta")
    for store in (model, words, events):
        store.reset(1)
    room, seen = 0, False
    errors = []

    def move(action):
        nonlocal room, seen
        key = cue(2 * room + action, 6)
        prediction = model.recall(key).copy()  # before the outcome is available
        room = (room + (1 if action == 0 else -1)) % 3  # environment executes
        observed = cue(room, 3)
        errors.append(float(np.square(prediction - observed).sum()))
        model.observe(key, observed)  # repair from the actual consequence
        if room == 2 and not seen:
            events.observe(cue(1, 3), observed)  # object 1 is visible only once
            seen = True

    for moment in range(64):
        move(int(rng.integers(2)))
        if moment in (10, 20):
            # Ambiguous scenes share object 1; no dictionary target is supplied.
            candidates = (0, 1) if moment == 10 else (1, 2)
            words.observe(cue(0, 1), np.maximum.reduce([cue(i, 3) for i in candidates]))
    while room != 0:
        move(0)

    # Read the same experience under interventions. These copies never teach the live agent.
    choices = {}
    for variant in ("full", "no_events", "no_words", "wrong_world"):
        world_copy, word_copy, event_copy = deepcopy((model, words, events))
        if variant == "no_events":
            event_copy.reset(1)
        elif variant == "no_words":
            word_copy.clear()
        elif variant == "wrong_world":
            world_copy.strength = np.roll(world_copy.strength, 1, axis=2)
        choices[variant] = request(world_copy, word_copy, event_copy, room)
    plan = choices["full"]
    for action in plan or ():
        move(action)  # acting and learning continue through the very same operation
    return {
        "choices": choices, "reached_object": room == 2,
        "first_16_prediction_error": float(np.mean(errors[:16])),
        "last_16_prediction_error": float(np.mean(errors[-16:])),
        "single_sightings": events.writes,
        "boundary": "Supplied perception, request protocol and search; learned associations.",
    }


if __name__ == "__main__":
    result = run()
    assert result["reached_object"] and result["single_sightings"] == 1
    assert result["choices"]["no_events"] is None and result["choices"]["no_words"] is None
    assert result["choices"]["wrong_world"] != result["choices"]["full"]
    assert result["last_16_prediction_error"] < result["first_16_prediction_error"]
    print(result)
