from utils.params import ParamsManager
from utils.cnn import CNN
from utils.framesPreprocessor import Preprocessor
from utils.memory import Memory
from utils.breakout import BreakoutWrapper
import numpy as np
import torch
import gym
from argparse import ArgumentParser
import copy
from agent import Agent


class BreakOutPlayer:
    def __init__(self, paramsManager):
        self.paramsManager = paramsManager
        self.memory = Memory(self.paramsManager.get_params()["agent"]["MEMORY_SIZE"],
                             self.paramsManager.get_params()["agent"]["MINI_BATCH_SIZE"],
                             self.paramsManager.get_params()["environment"]["FRAME_PROCESSED_WIDTH"],
                             self.paramsManager.get_params()["environment"]["FRAME_PROCESSED_HEIGHT"],
                             self.paramsManager.get_params()["environment"]["NUMBER_OF_FRAMES_TO_STACK_ON_STATE"])
        print("[i] Creating main convolutional neural network")
        self.main_cnn = CNN()
        print("[i] Creating target convolutional neural network")
        self.target_cnn = copy.deepcopy(self.main_cnn)
        print("[!] Creating the agent")
        self.main_cnn.cuda()
        self.target_cnn.cuda()
        self.agent = Agent(self.main_cnn, self.target_cnn, self.paramsManager.get_params()["agent"]["EPSILON_MAX"],
                           self.paramsManager.get_params()["agent"]["NUMBER_OF_FRAMES_TO_CONSTANT_EPSILON"],
                           self.paramsManager.get_params()["agent"]["FIRST_EPSILON_DECAY"],
                           self.paramsManager.get_params()["agent"]["FRAMES_TO_FIRST_EPSILON_DECAY"],
                           self.paramsManager.get_params()["agent"]["FINAL_EPSILON_VALUE"],
                           self.paramsManager.get_params()["agent"]["FRAMES_TO_FINAL_EPSILON"],
                           self.paramsManager.get_params()["agent"]["EXPLORATION_PROBABILITY_DURING_EVALUATION"],
                           self.paramsManager.get_params()["agent"]["LEARNING_RATE"])
        self.breakout_wrapper = BreakoutWrapper(self.paramsManager.get_params()["environment"]["NAME"],
                                                self.paramsManager.get_params()["agent"]["NO_OP_STEPS"],
                                                self.paramsManager.get_params()["environment"]["NUMBER_OF_FRAMES_TO_STACK_ON_STATE"],
                                                self.paramsManager.get_params()["environment"]["FRAME_PROCESSED_WIDTH"],
                                                self.paramsManager.get_params()["environment"]["FRAME_PROCESSED_HEIGHT"],
                                                self.paramsManager.get_params()["environment"]["RENDER"])

    def train(self):
        frame_number = 0
        rewards = []
        loss_list = []
        while frame_number < self.paramsManager.get_params()["agent"]["MAX_FRAMES"]:
            epoch = 0
            while epoch < self.paramsManager.get_params()["agent"]["EVAL_FREQUENCY"]:
                done_life_lost = self.breakout_wrapper.reset(evaluation=False)
                total_episode_reward = 0
                for i in range(self.paramsManager.get_params()["agent"]["MAX_EPISODE_LENGTH"]):
                    chosen_action = self.agent.get_action(frame_number, self.breakout_wrapper.actual_state, evaluation=False)
                    processed_new_frame, reward, done, done_life_lost, new_frame = self.breakout_wrapper.step(chosen_action)
                    if len(rewards) != 0:
                        print("Action performed: ", chosen_action, ". Reward: ", reward, ". Mean reward: ", sum(rewards)/len(rewards), ".Frame number: ", frame_number)
                    frame_number += 1
                    epoch += 1
                    total_episode_reward += reward
                    if self.paramsManager.get_params()["agent"]["CLIP_REWARD"]:
                        self.memory.store(processed_new_frame, chosen_action, self.clip_reward(reward), done_life_lost)
                    else:
                        self.memory.store(processed_new_frame, chosen_action, reward, done_life_lost)
                    # If its time to learn
                    if frame_number % self.paramsManager.get_params()["agent"]["UPDATE_FREQUENCY"] and frame_number > self.paramsManager.get_params()["agent"]["REPLAY_MEMORY_START_SIZE"]:
                        print("\n\n\n\n\n LEARNING")
                        losses = self.agent.learn(self.memory, self.paramsManager.get_params()["agent"]["GAMMA"])
                        print("Loss: ", losses)
                    if frame_number % self.paramsManager.get_params()["agent"]["NETWORK_UPDATE_FREQ"] == 0 and frame_number> self.paramsManager.get_params()["agent"]["REPLAY_MEMORY_START_SIZE"]:
                        self.agent.updateNetworks()
                    if done:
                        done = False
                        break
                rewards.append(total_episode_reward)




    def clip_reward(self, r):
        if r>0:
            return 1
        elif r==0:
            return 0
        else:
            return -1




if __name__ == '__main__':
    args = ArgumentParser("Breakout DeepQLearning")
    args.add_argument("--params-file",
                      help="Path to JSON Params File. Default: parameters.json",
                      default="parameters.json", metavar="PFILE")
    args = args.parse_args()
    paramsManager = ParamsManager(args.params_file)
    BreakoutPlayer = BreakOutPlayer(paramsManager)
    BreakoutPlayer.train()
