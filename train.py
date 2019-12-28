import gym

from agent import DQN

env = gym.make('LunarLander-v2')
dqn = DQN(env)


epochs = 1000
save_frequency = 25

e = 1
while epochs - e > 0:
    state = env.reset()
    done = False
    step = 0
    dqn.decrease_epsilon()

    if (epochs - e) % save_frequency == 0:
        dqn.save()
        print('Model saved')

    while not done:
        action = dqn.act(state)
        next_state, reward, done, info = env.step(action)
        dqn.remember(state, action, reward, next_state, done)
        dqn.replay()

        state = next_state
        step += reward

        if done:
            print("Epoch: " + str(e) + " Score: " + str(step) + " eps:" + str(dqn.epsilon))
            e += 1

env.close()
