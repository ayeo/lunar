import gym
from gym.wrappers import Monitor

from dqn import DQN

env = gym.make('LunarLander-v2')
#env = Monitor(env, 'videos', video_callable=lambda episode_id: True)
dqn = DQN(env)
dqn.load()


for x in range(20):
    state = env.reset()
    done = False

    while not done:
        env.render()
        action = dqn.act(state, False)
        next_state, reward, done, info = env.step(action)
        state = next_state

env.close()