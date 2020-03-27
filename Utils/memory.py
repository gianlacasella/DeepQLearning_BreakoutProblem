
import numpy as np
import random


class Memory:
    def __init__(self, memory_size, minibatch_size, width, height, frames_to_stack):
        print("[i] Initializing Memory with size ", memory_size, ", mini-batch size", minibatch_size,
              ", storing frames of ", width, "x", height, ", and each state is given by ", frames_to_stack, "frames")
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.frame_width, self.frame_height = width, height
        self.number_frames_to_stack = frames_to_stack
        self.number_of_memories = 0

        self.dones = np.empty(self.memory_size, dtype=np.bool)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.actions = np.empty(self.memory_size, dtype=np.int32)
        self.frames = np.empty((self.memory_size, self.frame_height, self.frame_width), dtype=np.uint8)

    def store(self, processed_frame, action, reward, done):
        if self.number_of_memories == self.memory_size:
            np.delete(self.actions, 0)
            np.delete(self.dones, 0)
            np.delete(self.rewards, 0)
            np.delete(self.frames, 0)
            self.number_of_memories = self.memory_size - 1
        self.actions[self.number_of_memories] = action
        self.dones[self.number_of_memories] = done
        self.rewards[self.number_of_memories] = reward
        self.frames[self.number_of_memories, ...] = processed_frame
        self.number_of_memories += 1

    def reset_memory(self):
        print("[i] Resetting Memory")
        self.dones = np.empty(self.memory_size, dtype=np.bool)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.actions = np.empty(self.memory_size, dtype=np.int32)
        self.frames = np.empty((self.memory_size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.number_of_memories = 0

    def get_minibatch(self):
        print("[i] Selecting mini-batch of size ", self.minibatch_size, " to replay training")
        states = np.empty((self.minibatch_size, self.number_frames_to_stack, self.frame_height, self.frame_width),
                          dtype=np.uint8)
        next_states = np.empty((self.minibatch_size, self.number_frames_to_stack, self.frame_height, self.frame_width),
                          dtype=np.uint8)

        memories_indexes = []
        for i in range(self.minibatch_size):
            index = random.randint(self.number_frames_to_stack, self.number_of_memories-1)
            if index not in memories_indexes:
                memories_indexes.append(index)

        print("[i] Selected memories indexes: ")
        for index in memories_indexes:
            print(index)

        returning_actions = np.empty(self.minibatch_size, dtype=np.int32)
        returning_rewards = np.empty(self.minibatch_size, dtype=np.float32)
        returning_dones = np.empty(self.minibatch_size, dtype=np.bool)

        count = 0
        for i in memories_indexes:
            states[count] = self.frames[(i-1)-self.number_frames_to_stack+1:i, ...]
            next_states[count] = self.frames[i-self.number_frames_to_stack+1:i+1, ...]
            returning_actions[count] = self.actions[count]
            returning_rewards[count] = self.rewards[count]
            returning_dones[count] = self.dones[count]
            count += 1

        return np.transpose(states, axes=(0, 2, 3, 1)), returning_actions, returning_rewards, \
               np.transpose(next_states, axes=(0, 2, 3, 1)), returning_dones
