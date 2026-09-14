"""Pure scheduling tests; POPGym is not needed."""

import unittest

import numpy as np
from delay import SensoryDelay


class TestSensoryDelay(unittest.TestCase):
    def test_actual_easy_lag_and_read_before_write(self):
        delay = SensoryDelay(2)
        tape = np.array([[0, 3], [1, 2], [2, 1], [3, 0], [0, 2]])
        for t, observation in enumerate(tape):
            before = delay.state.copy()
            read = delay.features(observation)
            np.testing.assert_array_equal(delay.state, before)
            expected = np.zeros((2, 4)) if t < 3 else np.eye(4)[tape[t - 3]]
            np.testing.assert_array_equal(read[:, 4:], expected)
            np.testing.assert_array_equal(delay.features(observation, True)[:, 4:], 0)
            delay.observe(observation)

    def test_future_observations_do_not_change_earlier_memory(self):
        original = [0, 1, 2, 3, 0, 1, 2]
        changed = [0, 1, 2, 0, 3, 2, 1]

        def reads(tape):
            delay = SensoryDelay(1)
            result = []
            for card in tape:
                result.append(delay.features(np.array([card]))[:, 4:])
                delay.observe(np.array([card]))
            return np.array(result)

        np.testing.assert_array_equal(reads(original)[:6], reads(changed)[:6])
        self.assertFalse(np.array_equal(reads(original)[6], reads(changed)[6]))
        np.testing.assert_array_equal(SensoryDelay(1).state, 0)


if __name__ == "__main__":
    unittest.main()
