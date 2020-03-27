from utils.params import ParamsManager
from utils.cnn import CNN
from utils.framesPreprocessor import framesProcessor
import numpy as np
import torch
import gym
from argparse import ArgumentParser


class BreakOutPlayer:
    def __init__(self, paramsManager):

        # to update target_cnn target_cnn.load_state_dict(main_cnn.state_dict())






if __name__ == '__main__':
    args = ArgumentParser("Breakout DeepQLearning")
    args.add_argument("--params-file",
                      help="Path to JSON Params File. Default: parameters.json",
                      default="parameters.json", metavar="PFILE")
    args = args.parse_args()
    paramsManager = ParamsManager(args.params_file)
    BreakoutPlayer = BreakOutPlayer(paramsManager)
