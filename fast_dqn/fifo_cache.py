from collections import deque


class FIFOCache:
    def __init__(self, maxsize):
        self._maxsize = maxsize
        self._dict = {}
        self._deque = deque()

    def __getitem__(self, state):
        state = self._hashable(state)
        return self._dict[state]

    def push(self, state, action):
        state = self._hashable(state)

        # Delete the oldest entry if cache is full
        if len(self._deque) == self._maxsize:
            old_state = self._deque.popleft()
            self._dict.pop(old_state)

        # Add new entry
        if state in self._dict:
            # If there's an older version, remove it
            self._deque.remove(state)
        # Overwrite
        self._dict[state] = action
        self._deque.append(state)

    def _hashable(self, state):
        return state.tobytes()
