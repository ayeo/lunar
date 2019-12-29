import gym

from dqn import DQN

env = gym.make('LunarLander-v2')
dqn = DQN(env)

# todo: implement early stop
epochs = 1000
save_frequency = 25

e = 1
while epochs - e > 0:
    state = env.reset()
    done = False
    score = 0

    if (epochs - e) % save_frequency == 0:
        dqn.save()
        print('Model saved')

    while not done:
        env.render()
        action = dqn.act(state)
        next_state, reward, done, info = env.step(action)
        dqn.remember(state, action, reward, next_state, done)
        dqn.replay()
        state = next_state
        score += reward

    dqn.adjust_epsilon()
    # todo: More important here is the average of few last episodes
    print("epoch: " + str(e) + " score: " + str(score) + " eps: " + str(dqn.epsilon))
    e += 1

env.close()
