"""One brain keeps learning while its environment changes. NumPy is the only dependency."""

import numpy as np

import cadence as cd


def run(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    agent = cd.GenericBrain.build(4, 4, hidden=16, seed=seed)
    context = rng.integers(4, size=32)
    action = agent.step(np.eye(4)[context])
    rewards = []
    for moment in range(600):
        # The world changes its reward rule halfway through the same learning life.
        rewarded_action = (context + int(moment >= 300)) % 4
        reward = (action == rewarded_action).astype(float)
        rewards.append(float(reward.mean()))
        context = rng.integers(4, size=32)
        action = agent.step(np.eye(4)[context], reward=reward, done=np.ones(32, bool))
    return {
        "before_change": float(np.mean(rewards[250:300])),
        "just_after_change": float(np.mean(rewards[300:310])),
        "after_adapting": float(np.mean(rewards[-50:])),
    }


if __name__ == "__main__":
    result = run()
    assert result["before_change"] > 0.9
    assert result["after_adapting"] > 0.9
    assert result["just_after_change"] < result["after_adapting"]
    print(result)
