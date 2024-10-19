import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3", render_mode='ansi')
env.reset()[0]

action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros(([state_size, action_size]))

episodes = 1000
epsilon = 0.01
alpha = 0.1
gamma = 0.9
hist = []


for episode in range(episodes):
    state = env.reset()[0]
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = np.random.randint(0, action_size)
            
        next_state, reward, done, truncated, info = env.step(action)
        
        current_value = qtable[state, action]  
        next_max = np.max(qtable[next_state]) 

        qtable[state, action] = (1 - alpha) * current_value + alpha * (reward + gamma * next_max)
        state = next_state

        

total_rewards = []
total_actions = []
for i in range(10):
    state = env.reset()[0]
    
    done = False
    total_reward = 0
    actions = 0

    while not done:
        print(env.render())
        action = np.argmax(qtable[state])
        next_state, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        actions += 1
        state = next_state

    total_rewards.append(total_reward)
    total_actions.append(actions)


avg_reward = np.mean(total_rewards)
avg_actions = np.mean(total_actions)
print(f"Average reward for 10 implementations: {avg_reward}")
print(f"Average actions for 10 implementations: {avg_actions}")

        
        