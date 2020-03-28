import gym
from tensorflow.keras.backend import resize_images
from tensorflow import image
import numpy as np
import random

class BreakoutWrapper:
    def __init__(self, environment_name, no_op_steps, frames_to_stack_on_state, frames_width, frames_height, render):
        print("[i] Creating the Breakout Wrapper on environment ", environment_name, ", ", frames_to_stack_on_state,
              " frames to stack on each state of size ", frames_height, "x", frames_width)
        self.env = gym.make(environment_name)
        self.processor = Preprocessor(frames_width, frames_height)
        self.actual_state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.frames_to_stack_on_state = frames_to_stack_on_state
        self.render = render


    def reset(self, evaluation):
        print("[i] Breakout Wrapper is resetting the environment")
        first_frame = self.env.reset()
        self.last_lives = 0
        done = True
        if evaluation:
            for i in range(random.randint(1, self.no_op_steps)):
                first_frame, a, b, c = self.env.step(1)
        processed_frame = self.processor.preprocessFrame(first_frame)
        self.actual_state = np.repeat(processed_frame, self.frames_to_stack_on_state, axis=2)
        return done

    def step(self, action):
        new_frame, reward, done, info = self.env.step(action)
        if self.render:
            self.env.render()
        if info['ale.lives'] < self.last_lives:
            done_life_lost = True
        else:
            done_life_lost = done
        self.last_lives = info['ale.lives']
        processed_new_frame = self.processor.preprocessFrame(new_frame)
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2)
        self.state = new_state

        return processed_new_frame, reward, done, done_life_lost, new_frame





# Transforms frames from 3x210x160 Tensors to 1x84x84 Tensor (84x84 grayscale picture)
class Preprocessor:
    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width

    def preprocessFrame(self, frame):
        # To grayscale
        self.returning_tensor = image.rgb_to_grayscale(frame)
        # Cropping
        self.returning_tensor = image.crop_to_bounding_box(self.returning_tensor, 34, 0, 160, 160)
        # Resizing to target_heightxtarget_width with nearest neighbor method
        self.returning_tensor = resize_images(self.returning_tensor, [self.target_height, self.target_width],
                                            image.ResizeMethod.NEAREST_NEIGHBOR)
        return self.returning_tensor
