import numpy as np
import random


class Memory:
    def __init__(self, total_stack_depth=1000000, batch_size=32, transition_stack_depth=4):
        self.total_stack_depth = total_stack_depth
        self.transition_stack_depth = transition_stack_depth
        self.batch_size = batch_size
        self.height, self.width = 84, 84
        self.count = 0
        self.current = 0

        self.actions = np.empty(self.total_stack_depth, dtype=np.int32)
        self.rewards = np.empty(self.total_stack_depth, dtype=np.float32)
        self.frames = np.empty((self.total_stack_depth, self.height, self.width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.total_stack_depth, dtype=np.bool)

        self.states = np.empty((self.batch_size, self.transition_stack_depth,
                                self.height, self.width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.transition_stack_depth,
                                    self.height, self.width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    # Returns the state at some index. A state is a set of 4 Tensors 84x84x1 (4 grayscale frames)
    def get_state(self, index):
        return self.frames[index - self.transition_stack_depth + 1:index + 1, ...]

    # Returns a minibatch from the experience
    def get_minibatch(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.transition_stack_depth, self.count - 1)
                if index < self.transition_stack_depth:
                    continue
                if index >= self.current and index - self.transition_stack_depth <= self.current:
                    continue
                if self.terminal_flags[index - self.transition_stack_depth:index].any():
                    continue
                break
            self.indices[i] = index

        for i, index in enumerate(self.indices):
            self.states[i] = self.get_state(index - 1)
            self.new_states[i] = self.get_state(index)

        # returns 32 stacks of experiences with format [state, action, reward, new_state, terminal]
        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[
            self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]

    def load_new_experience(self, reward, action, frame,  terminal):
        self.rewards[self.current] = reward
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.total_stack_depth
