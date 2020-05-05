# DeepQLearningBreakout


## Using DeepQLearning on the OpenAI's BreakoutDeterministic-v4 environment

On this project, I trained an AI Agent with **DeepQLearning (Double Deep Q-Learning with Dueling Network Architecture)** on the OpenAI's [BreakoutDeterministic-v4](https://gym.openai.com/envs/Breakout-v0/) environment.
<br><br>In Breakout, a layer of bricks lines the top third of the screen and the goal is **to destroy them all**. A ball moves straight around the screen, bouncing off the top and two sides of the screen. When a brick is hit, the ball bounces back and the brick is destroyed. The player loses a turn when the ball touches the bottom of the screen; to prevent this from happening, the player has a **horizontally movable paddle to bounce the ball upward, keeping it in play**. 

<p align="center">
  <img src="img/breakout.gif">
</p>

The algorithm used in this project is taken from [Mnih et al. 2015](https://www.nature.com/articles/nature14236/). Initially, it initializes replay memory with some capacity and creates two Neural Networks: action-value function with random weights and a target action-value function with the same weights. Then, it preprocesses a sequence of frames, and using an epsilon-greedy-descendant algorithm selects a random action or the best action selected by the action-value function. Executes the action on the environment, gets its reward and next image and then preprocesses the next frames sequence. Once done, stores the transition on the memory (a transition is given by a sequence, action performed, reward, and next sequence). 

<p align="left">
  <img src="img/algorithm.png">
</p>

Then, takes a "random minibatch" from the memory to replay, calculates the target network predicted q-value, and perform gradient descent step on the networks prediction difference with respect to the action-value function parameters. It is important to notice that the target neural network is predicting the q-value of the best action on the "next sequence".

At last, every C steps resets the target neural network.


## My results

## Prerequisites

## Getting started


## What I learned


## Authors

* **Gianfranco Lacasella** - *Initial work* - [glacasellaUANDES](https://github.com/glacasellaUANDES)

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE- see the [LICENSE.md](LICENSE.md) file for details
