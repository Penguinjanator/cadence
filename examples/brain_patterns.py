"""A supplied adversarial toy world: read candidate activity and reconsider futures."""

import numpy as np

from cadence.circuits import ActivityMonitor, imagine


def move(state: list[int], action: int) -> list[int]:
    state.append(action)
    return state


def value(state: list[int]) -> float:
    if len(state) < 2:
        return 0.0
    return (10.0 if state[1] == 0 else -10.0) if state[0] == 0 else 2.0


def run() -> int:
    live: list[int] = []
    monitor = ActivityMonitor()
    initial = imagine(live, lambda s: (0, 1), move, value, lambda s: len(s) == 2, depth=1)
    scores = np.array([f.score for f in initial.futures])
    readback = monitor.read(scores, scores)
    if readback.request_more:
        result = imagine(
            live, lambda s: (0, 1), move, value, lambda s: len(s) == 2, depth=2, adversarial=True
        )
    else:
        result = initial
    assert live == []
    return result.futures[0].action


if __name__ == "__main__":
    print("Self-monitor requested a deeper look; selected route:", run())
