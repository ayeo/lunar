import random
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


GAMMA = .75
ALPHA = .0002
EPSILON = .8
EPSILON_DECAY = .99995
EPSILON_MIN = .001

class DQN:
    def __init__(self, observations, actions, memory, batch):
        self.input = observations
        self.output = actions
        self.memory = deque(maxlen=memory)
        self.batch = batch

        self.epsilon = EPSILON

        self.model = Sequential()
        self.model.add(Dense(12, input_dim=observations, activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(actions, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=ALPHA))

    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.input])
        next_state = np.reshape(next_state, [1, self.input])
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, self.input])
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        if np.random.rand() < self.epsilon:
            return random.randrange(self.output)
        q_values = self.model.predict(state)

        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch:
            return
        for state, action, reward, next_state, done in random.sample(self.memory, self.batch):
            target = reward
            if not done:
                target = reward + GAMMA * np.max(self.model.predict(next_state)[0])


            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)


    def save(self, filename):
        self.model.save_weights(filename)

    def load(self, filename):
        self.model.load_weights(filename)
