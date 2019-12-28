import gym

from agent import DQN

env = gym.make('LunarLander-v2')
dqn = DQN(env)
dqn.load()


state = env.reset()
done = False

while not done:
    env.render()
    action = dqn.act(state, False)
    next_state, reward, done, info = env.step(action)
    state = next_state

env.close()