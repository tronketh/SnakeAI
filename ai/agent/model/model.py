import os

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
import numpy as np

import ai.constants as constants
from ai.agent.model import SAVED_MODEL_PATH


class GameAI:
    def __init__(self):
        inputs = Input(11, )
        x = inputs
        x = Dense(128, activation='leaky_relu')(x)
        x = Dense(32, activation='leaky_relu')(x)
        x = Dense(3, activation='leaky_relu')(x)
        outputs = x
        self.model = Model(inputs, outputs)
        self.optimizer = Adam(learning_rate=constants.LR, decay=constants.DECAY)
        self.loss_fn = MeanSquaredError()
        self.model.compile(self.optimizer, self.loss_fn)

    def model_train_predict(self, inputs, training=True):
        inputs = np.array(inputs).astype(float)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape([1, -1])
        return self.model(inputs, training=training)

    def train_step(self, old_state, move, reward, next_state, done):
        if len(np.array(next_state).shape) == 1:
            old_state = np.reshape(old_state, [1, -1])
            move = np.reshape(move, [1, -1])
            reward = np.reshape(reward, [1, -1])
            next_state = np.reshape(next_state, [1, -1])
            done = np.reshape(done, [1, -1])
        with tf.GradientTape() as tape:
            pred = self.model_train_predict(old_state)
            target = tf.identity(pred)
            target0 = np.array(target)
            for idx in range(len(done)):
                q_new = reward[idx]
                if not done[idx]:
                    q_new = reward[idx] + constants.GAMMA * tf.math.reduce_max(self.model_train_predict(next_state[idx]))
                move = np.array(move)
                target0[idx][np.argmax(move[idx]).item()] = q_new
            target = tf.convert_to_tensor(target0)
            loss = self.loss_fn(target, pred)
        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

    def save(self):
        file_name = SAVED_MODEL_PATH
        self.model.save(file_name)
        print('Model weights Saved', file_name)

    def load(self):
        file_name = SAVED_MODEL_PATH
        if os.path.exists(file_name):
            self.model = load_model(file_name)
            print('Model weights Loaded', file_name)
