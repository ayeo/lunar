import random
import numpy as np

from collections import deque
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

GAMMA = .99
ALPHA = .0001
EPSILON = .5
EPSILON_DECAY = .995
EPSILON_MIN = .01

FILENAME = 'model.h5'
MEMORY = 100000
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
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(self.actions, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=ALPHA))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def adjust_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def act(self, state, explore=True):
        state = np.reshape(state, [1, self.observations])
        if explore and np.random.rand() < self.epsilon:
            return random.randrange(self.actions)
        return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.batch:
            return
        states, actions, rewards, next_states, done_list = self.get_sample()
        targets = rewards + GAMMA * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        vector = self.model.predict_on_batch(states)
        indexes = np.array(range(self.batch))
        vector[indexes, actions] = targets
        self.model.fit(states, vector, epochs=1, verbose=0)

    def get_sample(self):
        #todo: how to convert list to ndarray?
        random_sample = random.sample(self.memory, self.batch)
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        return states, actions, rewards, next_states, done_list

    def save(self):
        self.model.save(self.filename, overwrite=True)

    def load(self):
        self.model = load_model(self.filename)
