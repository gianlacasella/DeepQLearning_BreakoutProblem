
import numpy as np
import random


class Memory:
    def __init__(self, good_memories_size, bad_memories_size, minibatch_size, width, height, frames_to_stack):
        print("[i] Initializing good Memory with size ", good_memories_size, ",bad Memory size ", bad_memories_size,
              ", mini-batch size", minibatch_size, ", storing frames of ", width, "x", height,
              ", and each state is given by ", frames_to_stack, "frames")
        self.good_memories_size = good_memories_size
        self.bad_memories_size = bad_memories_size
        self.minibatch_size = minibatch_size
        self.frame_width, self.frame_height = width, height
        self.number_frames_to_stack = frames_to_stack
        self.number_of_good_memories = 0
        self.number_of_bad_memories = 0
        self.current_worst_reward = 0
        self.current_best_reward = -np.inf

        self.good_dones = np.empty(self.good_memories_size, dtype=np.bool)
        self.good_rewards = np.empty(self.good_memories_size, dtype=np.float32)
        self.good_actions = np.empty(self.good_memories_size, dtype=np.int32)
        self.good_frames = np.empty((self.good_memories_size, self.frame_height, self.frame_width), dtype=np.uint8)

        self.bad_dones = np.empty(self.bad_memories_size, dtype=np.bool)
        self.bad_rewards = np.empty(self.bad_memories_size, dtype=np.float32)
        self.bad_actions = np.empty(self.bad_memories_size, dtype=np.int32)
        self.bad_frames = np.empty((self.bad_memories_size, self.frame_height, self.frame_width), dtype=np.uint8)

    def store(self, processed_frame, action, reward, done):
        if self.number_of_good_memories == self.good_memories_size:
            np.delete(self.good_actions, 0)
            np.delete(self.good_dones, 0)
            np.delete(self.good_rewards, 0)
            np.delete(self.good_frames, 0)
            self.number_of_good_memories = self.good_memories_size - 1
        elif self.number_of_bad_memories == self.bad_memories_size:
            np.delete(self.bad_actions, 0)
            np.delete(self.bad_dones, 0)
            np.delete(self.bad_rewards, 0)
            np.delete(self.bad_frames, 0)
            self.number_of_bad_memories = self.bad_memories_size - 1

        if reward > self.current_best_reward:
            self.current_best_reward = reward
            self.good_actions[self.number_of_good_memories] = action
            self.good_dones[self.number_of_good_memories] = done
            self.good_rewards[self.number_of_good_memories] = reward
            self.good_frames[self.number_of_good_memories, ...] = np.squeeze(processed_frame, 2)
            self.number_of_good_memories += 1
        elif reward < self.current_worst_reward:
            self.current_worst_reward = reward
            self.bad_actions[self.number_of_good_memories] = action
            self.bad_dones[self.number_of_good_memories] = done
            self.bad_rewards[self.number_of_good_memories] = reward
            self.bad_frames[self.number_of_good_memories, ...] = np.squeeze(processed_frame, 2)
            self.number_of_bad_memories += 1
        return


    def reset_memory(self):
        print("[i] Resetting Memory")
        self.good_dones = np.empty(self.good_memories_size, dtype=np.bool)
        self.good_rewards = np.empty(self.good_memories_size, dtype=np.float32)
        self.good_actions = np.empty(self.good_memories_size, dtype=np.int32)
        self.good_frames = np.empty((self.good_memories_size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.number_of_good_memories = 0
        self.bad_dones = np.empty(self.bad_memories_size, dtype=np.bool)
        self.bad_rewards = np.empty(self.bad_memories_size, dtype=np.float32)
        self.bad_actions = np.empty(self.bad_memories_size, dtype=np.int32)
        self.bad_frames = np.empty((self.bad_memories_size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.number_of_bad_memories = 0
        return

    def get_minibatch(self):
        print("[i] Selecting mini-batch of size ", self.minibatch_size, " to replay training")
        states = np.empty((self.minibatch_size, self.number_frames_to_stack, self.frame_height, self.frame_width),
                          dtype=np.uint8)
        next_states = np.empty((self.minibatch_size, self.number_frames_to_stack, self.frame_height, self.frame_width),
                          dtype=np.uint8)
        returning_actions = np.empty(self.minibatch_size, dtype=np.int32)
        returning_rewards = np.empty(self.minibatch_size, dtype=np.float32)
        returning_dones = np.empty(self.minibatch_size, dtype=np.bool)
        good_memories_indexes = []
        bad_memories_indexes = []

        # Selecting valid indexes for good and bad memories. 1/4 of the memory_batch_size will be filled up with
        # good memories, and the other 3/4 with bad memories
        i = 0
        for i in range(self.minibatch_size):
            if i < int(self.minibatch_size/4):
                index = random.randint(self.number_frames_to_stack, self.number_of_good_memories-1)
                if index not in good_memories_indexes:
                    good_memories_indexes.append(index)
            else:
                index = random.randint(self.number_frames_to_stack, self.number_of_bad_memories - 1)
                if index not in bad_memories_indexes:
                    bad_memories_indexes.append(index)
        # Adding memories to the batch
        count = 0
        for i in self.minibatch_size:
            if i < self.minibatch_size/2:
                states[count] = self.good_frames[(i-1)-self.number_frames_to_stack+1:i, ...]
                next_states[count] = self.good_frames[i-self.number_frames_to_stack+1:i+1, ...]
                returning_actions[count] = self.good_actions[count]
                returning_rewards[count] = self.good_rewards[count]
                returning_dones[count] = self.good_dones[count]
            else:
                states[count] = self.bad_frames[(i - 1) - self.number_frames_to_stack + 1:i, ...]
                next_states[count] = self.bad_frames[i - self.number_frames_to_stack + 1:i + 1, ...]
                returning_actions[count] = self.bad_actions[count]
                returning_rewards[count] = self.bad_rewards[count]
                returning_dones[count] = self.bad_dones[count]
            count += 1
        # Returning a tuple where each element is a numpy array with memory data
        return np.transpose(states, axes=(0, 2, 3, 1)), returning_actions, returning_rewards, \
               np.transpose(next_states, axes=(0, 2, 3, 1)), returning_dones
