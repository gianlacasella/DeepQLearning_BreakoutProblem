from utils.params import ParamsManager
from utils.memory import Memory
from utils.cnn import CNN
from utils.env import EnvironmentManager
from utils.agent import Agent
import numpy as np
import torch
import gym

PARAMS_FILE = "params.json"


class BreakOutPlayer:
    def __init__(self):
        self.paramsManager = ParamsManager(PARAMS_FILE)
        self.memory = Memory(total_stack_depth=self.paramsManager.get_params()["total_stack_depth"],
                        batch_size=self.paramsManager.get_params()["batch_size"],
                        transition_stack_depth=self.paramsManager.get_params()["transition_stack_depth"])
        self.main_cnn = CNN(self.paramsManager.get_params()["device"])
        self.target_cnn = CNN(self.paramsManager.get_params()["device"])
        self.environment_manager = EnvironmentManager()
        self.agent = Agent(self.main_cnn, self.paramsManager.get_params()['epsilon_initial'],
                           self.paramsManager.get_params()['epsilon_final'],
                           self.paramsManager.get_params()['max_steps'])
        # to update target_cnn target_cnn.load_state_dict(main_cnn.state_dict())

    def train(self):
        frame_number = 0
        rewards = []
        loss_list = []
        while frame_number < self.paramsManager.get_params()["MAX_FRAMES"]:
            epoch_frame = 0
            # Eval_frequency describes how many times we stop training to evaluate performance
            while epoch_frame < self.paramsManager.get_params()["EVAL_FREQUENCY"]:
                terminal_life_lost = self.environment_manager.reset()
                reward_sum = 0
                for step in self.paramsManager.get_params()["MAX_EP_LENGTH"]:
                    action = self.agent.get_action(self.environment_manager.state, epoch_frame)
                    new_frame, reward, terminal, terminal_life_lost, i = self.environment_manager.step(action)
                    frame_number +=1
                    epoch_frame += 1
                    reward_sum += reward

                    reward = clip_reward(reward)

                    self.memory.load_new_experience(reward, action, frame=new_frame[:, :, 0], terminal=terminal_life_lost)
                    if frame_number % self.paramsManager.get_params()['replay_freq'] and frame_number > self.paramsManager.get_params()['replay_start_size']:
                        loss = learn(self.memory, self.main_cnn, self.target_cnn, self.paramsManager.get_params()["batch_size"],
                                     self.paramsManager.get_params()['gamma'])
                        loss_list.append(loss)



def learn(memory, main_cnn, target_cnn, batch_size, gamma):
    states, actions, rewards, new_states, terminal_flags = memory.get_minibatch()
    for new_state in new_states:
        arg_q_max = np.argmax(main_cnn(new_state).data.to(torch.device('cpu')).numpy())
        q_vals = target_cnn(new_state).data.to(torch.device('cpu')).numpy()
        #...


def clip_reward(r):
    if r>0:
        return 1
    elif r==0:
        return 0
    else:
        return -1








if __name__ == '__main__':
    BreakoutPlayer = BreakOutPlayer()
    BreakOutPlayer.train()