import gym
import numpy as np

from dqn import DQN

target = 270
sample_size = 50

env = gym.make('LunarLander-v2')
dqn = DQN(env)

e = 1
scores_list = []
while True:
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = dqn.act(state)
        next_state, reward, done, info = env.step(action)
        dqn.remember(state, action, reward, next_state, done)
        dqn.replay()
        state = next_state
        score += reward

    scores_list.append(score)
    last_rewards_mean = np.mean(scores_list[sample_size * -1:])
    print(
        str(e) + ": \t\t" +
        str(round(float(score))) + "\t\t" +
        str(round(float(last_rewards_mean))) + "\t\t" +
        str(round(dqn.epsilon, 3))
    )

    dqn.adjust_epsilon()
    e += 1
    if last_rewards_mean > target:
        print("DQN trained")
        break

dqn.save()
env.close()
