from utils.params import ParamsManager
from utils.cnn import CNN
from utils.framesPreprocessor import Preprocessor
from utils.memory import Memory
import numpy as np
import torch
import gym
from argparse import ArgumentParser
import copy
from agent import Agent


# to update target_cnn target_cnn.load_state_dict(main_cnn.state_dict())

class BreakOutPlayer:
    def __init__(self, paramsManager):
        self.paramsManager = paramsManager
        self.memory = Memory(self.paramsManager.get_params()["agent"]["MEMORY_SIZE"],
                             self.paramsManager.get_params()["agent"]["MINI_BATCH_SIZE"],
                             self.paramsManager.get_params()["environment"]["FRAME_PROCESSED_WIDTH"],
                             self.paramsManager.get_params()["environment"]["FRAME_PROCESSED_HEIGHT"],
                             self.paramsManager.get_params()["environment"]["NUMBER_OF_FRAMES_TO_STACK_ON_STATE"])
        print("[i] Creating main convolutional neural network")
        self.main_cnn = CNN("cuda" if self.paramsManager.get_params()["agent"]["USE_CUDA"]==True else "cpu")
        print("[i] Creating target convolutional neural network")
        self.target_cnn = copy.deepcopy(self.main_cnn)
        print("[!] Creating the agent")
        self.agent = Agent(self.main_cnn, self.target_cnn, self.paramsManager.get_params()["agent"]["EPSILON_MAX"],
                           self.paramsManager.get_params()["agent"]["NUMBER_OF_FRAMES_TO_CONSTANT_EPSILON"],
                           self.paramsManager.get_params()["agent"]["FIRST_EPSILON_DECAY"],
                           self.paramsManager.get_params()["agent"]["FRAMES_TO_FIRST_EPSILON_DECAY"],
                           self.paramsManager.get_params()["agent"]["FINAL_EPSILON_VALUE"],
                           self.paramsManager.get_params()["agent"]["FRAMES_TO_FINAL_EPSILON"],
                           self.paramsManager.get_params()["agent"]["EXPLORATION_PROBABILITY_DURING_EVALUATION"])



if __name__ == '__main__':
    args = ArgumentParser("Breakout DeepQLearning")
    args.add_argument("--params-file",
                      help="Path to JSON Params File. Default: parameters.json",
                      default="parameters.json", metavar="PFILE")
    args = args.parse_args()
    paramsManager = ParamsManager(args.params_file)
    BreakoutPlayer = BreakOutPlayer(paramsManager)
