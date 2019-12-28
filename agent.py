import random
import numpy as np

from collections import deque
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

GAMMA = .99
ALPHA = .0001
EPSILON = 1.0
EPSILON_DECAY = .995
EPSILON_MIN = .001

FILENAME = 'model.h5'
MEMORY = 500000
BATCH = 32

class DQN:
    def __init__(self, env):
        self.filename = FILENAME
        self.observations = env.observation_space.shape[0]
        self.actions = env.action_space.n
        self.build_model()

    def build_model(self):
        self.memory = deque(maxlen=MEMORY)
        self.batch = BATCH
        self.epsilon = EPSILON
        self.model = Sequential()
        self.model.add(Dense(512, input_dim=self.observations, activation="relu"))
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dense(self.actions, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=ALPHA))

    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.observations])
        next_state = np.reshape(next_state, [1, self.observations])
        self.memory.append((state, action, reward, next_state, done))

    def decrease_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def act(self, state, explore=True):
        state = np.reshape(state, [1, self.observations])
        if explore and np.random.rand() < self.epsilon:
            return random.randrange(self.actions)
        q_values = self.model.predict(state)

        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch:
            return

        for state, action, reward, next_state, done in random.sample(self.memory, self.batch):
            target = 0
            if not done:
                target = reward + GAMMA * np.max(self.model.predict(next_state)[0])


            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def save(self):
        self.model.save(self.filename, overwrite=True)

    def load(self):
        self.model = load_model(self.filename)
