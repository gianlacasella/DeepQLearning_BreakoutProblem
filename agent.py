import numpy as np
import torch
import copy

class Agent:
    def __init__(self, main_cnn, target_cnn, starting_epsilon, number_of_frames_to_constant_starting_epsilon,
                 first_epsilon_decay, number_of_frames_to_first_epsilon_decay, final_epsilon,
                 frames_to_final_epsilon, exploration_prob_during_eval, learning_rate):
        """
        :param main_cnn: Main convolutional neural network
        :param starting_epsilon: Starting epsilon value, usually 1
        :param number_of_frames_to_constant_starting_epsilon: Number of frames where epsilon will be constant when
        training starts
        :param first_epsilon_decay: First epsilon decay value
        :param number_of_frames_to_first_epsilon_decay: How many frames are needed to get to the first_epsilon_decay?
        :param final_epsilon: Final epsilon value
        :param max_frames: Max number of frames the agent will see
        :param exploration_prob_during_eval: Probability of exploration during evaluation
        """
        self.main_cnn = main_cnn
        self.target_cnn = target_cnn
        self.starting_epsilon = starting_epsilon
        self.number_frames_with_constant_epsilon = number_of_frames_to_constant_starting_epsilon
        self.first_epsilon_decay = first_epsilon_decay
        self.frames_to_first_decay = number_of_frames_to_first_epsilon_decay
        self.final_epsilon = final_epsilon
        self.frames_to_final_epsilon = frames_to_final_epsilon
        self.exploration_during_evaluation = exploration_prob_during_eval
        self.learning_rate = learning_rate

        self.first_slope = (self.first_epsilon_decay - self.starting_epsilon) / self.frames_to_first_decay
        self.first_intercept = -self.first_slope*self.number_frames_with_constant_epsilon + self.starting_epsilon
        self.second_slope = (self.final_epsilon - self.first_epsilon_decay) / self.frames_to_final_epsilon
        self.second_intercept = -self.second_slope*(self.number_frames_with_constant_epsilon + self.frames_to_first_decay) + self.first_epsilon_decay

        self.print_data()

        self.main_cnn_optimizer = torch.optim.Adam(self.main_cnn.parameters(), lr=self.learning_rate)

    def print_data(self):
        print("[i] Agent creation parameters: ")
        print("     [1] Starting epsilon: ", self.starting_epsilon)
        print("     [2] Number of frames with constant epsilon: ", self.number_frames_with_constant_epsilon)
        print("     [3] First epsilon decay: ", self.first_epsilon_decay)
        print("     [4] Frames to first decay: ", self.frames_to_first_decay)
        print("     [5] Final epsilon: ", self.final_epsilon)
        print("     [6] Frames to final epsilon: ", self.frames_to_final_epsilon)
        print("     [7] Exploration probability during evaluation: ", self.exploration_during_evaluation)
        print("     [8] First decay slope: ", self.first_slope)
        print("     [9] First intercept: ", self.first_intercept)
        print("     [10] Second decay slope: ", self.second_slope)
        print("     [11] Second intercept: ", self.second_intercept)

    def get_action(self, frame_number, state, evaluation):
        epsilon = 0
        if evaluation:
            epsilon = self.exploration_during_evaluation
            print("[i] Due to evaluation mode, epsilon is ", epsilon)
        elif frame_number < self.number_frames_with_constant_epsilon:
            epsilon = self.starting_epsilon
            print("[i] Frame inside the number of frames with constant epsilon, epsilon is ", epsilon)
        elif self.number_frames_with_constant_epsilon<frame_number < self.frames_to_first_decay:
            epsilon = frame_number*self.first_slope+self.first_intercept
            print("[i] Frame number inside the first decay period, epsilon is ", epsilon)
        elif self.frames_to_first_decay < frame_number < self.frames_to_final_epsilon:
            epsilon = frame_number*self.second_slope+self.second_intercept
            print("[i] Frame number inside the second decay period, epsilon is ", epsilon)
        if np.random.rand(1) < epsilon:
            chosen = np.random.randint(0, 3)
            # USED ACTIONS: [NOOP, LEFT, RIGHT]
            # OPENAI GYM ACTIONSPACE: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
            if chosen == 1:
                # Returns left
                print("[i] Random action chosen: LEFT")
                return 3
            elif chosen == 2:
                # Returns right
                print("[i] Random action chosen: RIGHT")
                return 2
            # Returns noop
            print("[i] Random action chosen: NOOP")
            return chosen
        best_action = np.argmax(self.main_cnn(state).data.to(torch.device('cpu')).numpy())
        if best_action == 1:
            print("[i] Best action chosen: LEFT")
            return 3
        elif best_action == 2:
            print("[i] Random action chosen: RIGHT")
            return 2
        print("[i] Random action chosen: NOOP")
        return best_action

    def learn(self, memory, gamma, mini_batch_size):
        states, actions, rewards, new_states, dones = memory.get_minibatch()
        losses = []
        for i in range(mini_batch_size):
            new_state = new_states[i]
            y = rewards[i] + \
                gamma * torch.max(self.target_cnn(new_state)) * \
                (1 - dones[i])
            Q = self.main_cnn(states[i])[actions[i]]
            loss = torch.nn.functional.smooth_l1_loss(Q, y)
            self.main_cnn_optimizer.zero_grad()
            loss.backward()
            self.main_cnn_optimizer.step()
            i += 1
            losses.append(loss)
        return losses

    def updateNetworks(self):
        self.target_cnn = copy.deepcopy(self.main_cnn)

