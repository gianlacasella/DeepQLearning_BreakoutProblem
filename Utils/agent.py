import random
import numpy as np
import torch


class Agent:
    def __init__(self, main_cnn, epsilon_initial_value, epsilon_final_value, max_steps):
        self.cnn = main_cnn
        self.epsilon_initial_value = epsilon_initial_value
        self.epsilon_final_value = epsilon_final_value
        self.max_steps = max_steps
        self.decay_factor = (self.epsilon_initial_value - self.epsilon_final_value) / self.max_steps

    def get_action(self, state, step_num):
        if random.random() < self.epsilon_decay(step_num):
            print("Random action chosen with epsilon: ", self.epsilon_decay(step_num))
            action = random.choice([a for a in range(4)])
        else:
            print("Greedy action chosen with epsilon: ", self.epsilon_decay(step_num))
            action = np.argmax(self.cnn(state).data.to(torch.device('cpu')).numpy())
        return action

    def epsilon_decay(self, step_number):
        current_value = self.epsilon_initial_value - step_number * self.decay_factor
        if current_value < self.epsilon_final_value:
            current_value = self.epsilon_final_value
        return current_value
