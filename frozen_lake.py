import time
import pickle
import os

import gym
import numpy as np

class FrozenLake:
    def __init__(self, 
                is_slippery=False,
                epsilon=1.0, 
                epsilon_decay=0.9993,
                total_episodes=20000,
                max_steps=100,
                lr_rate=0.8,
                gamma=0.95,
                learning='q'):
        self.is_slippery=is_slippery
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.total_episodes=total_episodes
        self.max_steps=max_steps
        self.lr_rate=lr_rate
        self.gamma=gamma
        self.learning=learning

        if not os.path.exists(os.path.dirname("~/opts/ml/model/")):
            try:
                os.makedirs(os.path.dirname("~/opts/ml/model/"))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        self.env = gym.make('FrozenLake-v0',is_slippery=self.is_slippery)
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
    
    def choose_action(self, state):
        action=0
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        return action

    def q_learning(self, state, state2, reward, action):
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = self.Q[state, action] + self.lr_rate * (target - predict)

    def sarsa_learning(self, state, state2, reward, action, action2):
        predict = self.Q[state, action]
        target = reward + self.gamma * self.Q[state2, action2]
        self.Q[state, action] = self.Q[state, action] + self.lr_rate * (target - predict)

    def test_game(self):
        reward_games = []
        for _ in range(100):
            state = self.env.reset()
            t = 0
            rewards = 0
            while t < self.max_steps:
                # Act greedly 
                action = self.choose_action(state) 
                state2, reward, done, info = self.env.step(action)
                state = state2
                rewards += reward

                t += 1

                if done:
                    reward_games.append(rewards)
                    break
    
        return np.mean(reward_games)

    def train(self):
        for episode in range(self.total_episodes):
            state = self.env.reset()
            t = 0
            self.epsilon = self.epsilon * self.epsilon_decay

            while t < self.max_steps:

                action = self.choose_action(state)  
                state2, reward, done, info = self.env.step(action)

                if self.learning == 'q':
                    self.q_learning(state, state2, reward, action)
                elif self.learning == 'sarsa':
                    action2 = self.choose_action(state2) 
                    self.sarsa_learning(state, state2, reward, action, action2)

                state = state2

                t += 1
            
                if done:
                    # Test the new table every 1k games
                    if episode % 1000 == 0:
                        test_reward = self.test_game()
                        print('\tEp:', episode, 'Test reward:', test_reward, np.round(self.epsilon,2))

                    break

        if self.learning == 'q':
            with open("~/opts/ml/model/frozenLake_qTable.pkl", 'wb') as f:
                pickle.dump(self.Q, f)
        elif self.learning == 'sarsa':
            with open("~/opts/ml/model/frozenLake_qTable_sarsa.pkl", 'wb') as f:
                pickle.dump(self.Q, f)

    def play(self):
        self.epsilon = 0
        if self.learning == 'q':
            with open("frozenLake_qTable.pkl", 'rb') as f:
                self.Q = pickle.load(f)            
        elif self.learning == 'sarsa':
            with open("frozenLake_qTable_sarsa.pkl", 'rb') as f:
                self.Q = pickle.load(f)            

        for episode in range(5):
            state = self.env.reset()
            print("*** Episode: ", episode)
            t = 0
            while t < self.max_steps:
                self.env.render()

                action = self.choose_action(state)  
                state2, reward, done, info = self.env.step(action)  
                state = state2

                if done:
                    break

                time.sleep(0.5)
                os.system('clear')

if __name__ == '__main__':
    frozen_lake_q_learning = FrozenLake(learning='q',is_slippery=True)
    frozen_lake_q_learning.train()
    # frozen_lake_q_learning.play()

