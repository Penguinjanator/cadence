"""Task-specific causal sensory delay; NumPy only, independent of POPGym."""

import numpy as np


class SensoryDelay:
    """Three internal one-step stages. Read first; write the current observation later."""

    def __init__(self, batch, lag=3):
        self.state = np.zeros((batch, lag, 4))

    def features(self, observation, masked=False):
        old = self.state[:, -1].copy()
        if masked:
            old[:] = 0
        return np.concatenate([np.eye(4)[observation], old], axis=1)

    def observe(self, observation):
        self.state[:, 1:] = self.state[:, :-1].copy()
        self.state[:, 0] = np.eye(4)[observation]
