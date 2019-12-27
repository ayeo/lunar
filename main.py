import gym

from agent import DQN

filename = 'weights.h5'

env = gym.make('LunarLander-v2')
dqn = DQN(observations=8, actions=4, memory=1000000, batch=30)
#dqn.load(filename)
epochs = 10000
save_frequency = 100

e = 0
while epochs - e > 0:
    state = env.reset()
    done = False
    step = 0

    if (epochs - e) % save_frequency == 0:
        dqn.save(filename)
        print('Model saved')

    while not done:
        #env.render()
        action = dqn.act(state)
        next_state, reward, done, info = env.step(action)
        dqn.remember(state, action, reward, next_state, done)
        dqn.replay()

        state = next_state
        step += reward

        if done:
            print("Epoch: " + str(e) + " Score: " + str(step) + " eps:" + str(dqn.epsilon))
            e += 1
            break


env.close()
