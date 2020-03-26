
import gym
from framesPreprocessor import Preprocessor
import numpy as np
import random

class EnvironmentManager:
    def __init__(self):
        self.environment = gym.make("'BreakoutDeterministic-v4'")
        self.processor = Preprocessor()
        self.actual_state = None
        self.last_lives = 0
        self.no_steps = 10
        self.agent_stack_len = 4

    def reset_environment(self):
        self.firstFrame = self.environment.reset()
        self.last_lives = 0
        terminal_life_lost = True
        if eval:
            for i in range(random.randint(1, self.no_steps)):
                frame, a, b, c = self.environment.step(1)
        processed_frame = self.processor.preprocessFrame(self, self.firstFrame)
        self.actual_state = np.repeat(processed_frame, self.agent_stack_len, axis=2)
        return terminal_life_lost

    def make_step(self, action):
        # Takes the step
        new_frame, reward, done, info = self.environment.step(action)
        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = done
        self.last_lives = info['ale.lives']
        # Processes new_frame
        processed_new_frame = self.processor.preprocessFrame(new_frame)
        # Creates the new state appending the new_frame to the actual_state
        new_state = np.append(self.actual_state[:, :, 1:], processed_new_frame, axis=2)
        # Updates the actual_state
        self.actual_state = new_state
        # Returns the result of making the step
        return processed_new_frame, reward, done, terminal_life_lost, new_frame
