from collections import deque

import numpy as np


class ImageStacker:
    def __init__(self, history_len):
        assert history_len >= 1
        self._history_len = history_len
        self._deque = deque(maxlen=history_len)
        self._shape = None
        self._dtype = None

    def append(self, observation, reset=False):
        if self._shape is None:
            self._shape = observation.shape
            self._dtype = observation.dtype

        if reset:
            self._reset()

        assert observation.shape == self._shape
        assert observation.dtype == self._dtype
        self._deque.append(observation)

    def get_stack(self):
        assert len(self._deque) == self._history_len
        return np.concatenate(list(self._deque), axis=-1)

    def _reset(self):
        for _ in range(self._history_len):
            self._deque.append(
                np.zeros(shape=self._shape, dtype=self._dtype))
