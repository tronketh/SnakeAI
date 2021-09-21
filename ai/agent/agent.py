import random

import numpy as np
from game.snake_game import SnakeGame
from ai.agent.model.model import GameAI
from collections import deque

import matplotlib.pyplot as plt

MAX_MEMORY = 100_000
BATCH_SIZE = 1000


class Agent:
    def __init__(self):
        self.game = SnakeGame()
        self.game_ai = GameAI()
        # self.game_ai.load()

        self.n_games = 0
        self.epsilon = 0  # randomness

        self.memory = deque(maxlen=MAX_MEMORY)

    def train_short_memory(self, state, move, reward, next_state, done):
        self.game_ai.train_step(state, move, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, moves, rewards, next_states, dones = zip(*mini_sample)
        self.game_ai.train_step(states, moves, rewards, next_states, dones)

    def remember(self, state, move, reward, next_state, done):
        self.memory.append((state, move, reward, next_state, done))

    def get_move(self, state, game_limit = 80):
        self.epsilon = game_limit - self.n_games
        final_move = np.array([0, 0, 0])
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = state.copy()
            prediction = np.array(self.game_ai.model_train_predict(state0, training=False))
            max_idx = np.argmax(prediction)
            final_move[max_idx] = 1
        return final_move.astype(float)

    def plot_live_data(self, scores, rewards):
        plt.axis([0, len(scores), -100, 100])
        plt.plot(rewards, color='blue')
        plt.pause(0.001)

    def play_train(self):
        self.game_ai.load()
        record = 0
        while True:
            state_old = self.game.get_state()
            final_move = self.get_move(state_old)
            reward, done, score = self.game.play_ai(final_move)

            state_new = self.game.get_state()
            self.train_short_memory(state_old, final_move, reward, state_new, done)
            self.remember(state_old, final_move, reward, state_new, done)

            if done:
                self.game.reset()
                self.n_games += 1
                self.train_long_memory()

                if score > record:
                    record = score
                    self.game_ai.save()
                if score == 1e10:
                    quit()

                print('game', self.n_games, 'Record:', record)

    def play_no_train(self):
        model = GameAI()
        model.load()
        record = 0
        while True:
            state_old = self.game.get_state()
            final_move = self.get_move(state_old)
            reward, done, score = self.game.play_ai(final_move)
            if done:
                self.game.reset()
                self.n_games += 1
                if score > record:
                    record = score
                if score == 1e10:
                    quit()
                print('game', self.n_games, 'Record:', record)
