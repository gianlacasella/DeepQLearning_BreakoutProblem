from utils.params import ParamsManager
from utils.cnn import CNN
from utils.framesPreprocessor import framesProcessor
from utils.memory import Memory
import numpy as np
import torch
import gym
from argparse import ArgumentParser


# to update target_cnn target_cnn.load_state_dict(main_cnn.state_dict())

class BreakOutPlayer:
    def __init__(self, paramsManager):
        self.paramsManager = paramsManager
        self.memory = Memory(self.paramsManager.get_params()["agent"]["MEMORY_SIZE"],
                             self.paramsManager.get_params()["agent"]["MINI_BATCH_SIZE"],
                             self.paramsManager.get_params()["environment"]["FRAME_PROCESSED_WIDTH"],
                             self.paramsManager.get_params()["environment"]["FRAME_PROCESSED_HEIGHT"],
                             self.paramsManager.get_params()["environment"]["NUMBER_OF_FRAMES_TO_STACK_ON_STATE"])







if __name__ == '__main__':
    args = ArgumentParser("Breakout DeepQLearning")
    args.add_argument("--params-file",
                      help="Path to JSON Params File. Default: parameters.json",
                      default="parameters.json", metavar="PFILE")
    args = args.parse_args()
    paramsManager = ParamsManager(args.params_file)
    BreakoutPlayer = BreakOutPlayer(paramsManager)
