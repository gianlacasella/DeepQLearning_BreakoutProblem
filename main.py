from utils.params import ParamsManager
from utils.cnn import CNN
from utils.memory import Memory
from utils.breakout import BreakoutWrapper
import numpy as np
import torch
import gym
from argparse import ArgumentParser
import copy
from agent import Agent
import time

class BreakOutPlayer:
    def __init__(self, paramsManager):
        self.paramsManager = paramsManager
        self.memory = Memory(self.paramsManager.get_params()["agent"]["GOOD_MEMORIES_SIZE"],
                             self.paramsManager.get_params()["agent"]["BAD_MEMORIES_SIZE"],
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
                           self.paramsManager.get_params()["agent"]["NUMBER_OF_FRAMES_WITH_CONSTANT_EPSILON"],
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
        # Stores the mean rewards of each epoch
        epochs_means = []
        # While we are training
        while frame_number < self.paramsManager.get_params()["agent"]["MAX_FRAMES"]:
            #########################
            ####### TRAINING ########
            #########################
            # Epoch counter
            epoch_counter = 0
            # Stores the epoch rewards
            epoch_rewards = []
            # While we arent on evaluation
            while epoch_counter < self.paramsManager.get_params()["agent"]["EVAL_FREQUENCY"]:
                # Resetting the env
                done_life_lost = self.breakout_wrapper.reset(evaluation=False)
                # Other params
                total_episode_reward = 0
                current_ale_lives = 5
                perform_fire = True
                for i in range(self.paramsManager.get_params()["agent"]["MAX_EPISODE_LENGTH"]):
                    # Prints the saparetor defined on the json
                    print(self.paramsManager.get_params()["environment"]["SEPARATOR"])
                    # If its necessary to FIRE
                    if perform_fire:
                        chosen_action = 1
                    else:
                        chosen_action = self.agent.get_action(frame_number, self.breakout_wrapper.actual_state,evaluation=False)
                    # We take the step. A dying penalty is added by the breakout_wrapper
                    processed_new_frame, reward, done, done_life_lost, _ , info = self.breakout_wrapper.step(chosen_action,self.paramsManager.get_params()["agent"]["DYING_REWARD"], current_ale_lives)
                    print("[i] Action performed: ", chosen_action, ". Reward: ", reward, ".Frame number: ", frame_number)
                    # If we already have rewards:
                    if len(rewards) != 0:
                        print("[i] Mean Training Reward: %.3f" % (sum(rewards)/len(rewards)))
                    if len(epoch_rewards) != 0:
                        print("[i] Mean Epoch Reward: %.3f" % (sum(epoch_rewards)/len(epoch_rewards)))
                    frame_number += 1
                    epoch_counter += 1
                    total_episode_reward += reward
                    if self.paramsManager.get_params()["agent"]["CLIP_REWARD"]:
                        self.memory.store(processed_new_frame, chosen_action, self.clip_reward(reward), done_life_lost)
                    else:
                        self.memory.store(processed_new_frame, chosen_action, reward, done_life_lost)
                    # If its time to learn
                    if frame_number % self.paramsManager.get_params()["agent"]["UPDATE_FREQUENCY"] and frame_number > self.paramsManager.get_params()["agent"]["REPLAY_MEMORY_START_FRAME"]:
                        losses = self.agent.learn(self.memory, self.paramsManager.get_params()["agent"]["GAMMA"], self.paramsManager.get_params()["agent"]["MINI_BATCH_SIZE"])
                    if frame_number % self.paramsManager.get_params()["agent"]["NETWORK_UPDATE_FREQ"] == 0 and frame_number> self.paramsManager.get_params()["agent"]["REPLAY_MEMORY_START_FRAME"]:
                        self.agent.updateNetworks()
                    if info["ale.lives"] < current_ale_lives:
                        perform_fire = True
                        current_ale_lives = info["ale.lives"]
                    elif info["ale.lives"] == current_ale_lives:
                        perform_fire = False
                    if done:
                        done = False
                        perform_fire = True
                        break
                rewards.append(total_episode_reward)
                epoch_rewards.append(total_episode_reward)
            #########################
            ####### SAVE INFO #######
            #########################
            epochs_means.append(sum(epoch_rewards)/len(epoch_rewards))
            file = open("results.txt", "w")
            print("============ EPOCH %d FINISHED ============"% len(epochs_means))
            for idx, mean in enumerate(epochs_means):
                print("Epoch number: %d. Mean reward: %.3f" % (idx, mean))
                file.write("Epoch number: %d. Mean reward: %.3f\n" % (idx, mean))
            file.close()
            time.sleep(10)
            


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
    if paramsManager.get_params()["agent"]["TRAINING"]:
        BreakoutPlayer.train()
