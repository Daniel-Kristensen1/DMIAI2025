from src.game.env import RaceCarEnv
from src.game.agent.dqn_agent import DQNAgent
from tqdm import trange
import torch

env = RaceCarEnv()
obs_dim = len(env.reset())
action_dim = 5  # Number of actions

agent = DQNAgent(obs_dim, action_dim)

num_episodes = 100
target_update_freq = 10

for episode in trange(num_episodes,desc="Training Progress"):
    obs = env.reset()
    done = False
    total_reward = 0
    max_steps = 1000
    step_count = 0
    while not done and step_count < max_steps:
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, done)
        agent.learn()
        obs = next_obs
        total_reward += reward
        step_count += 1

    print(step_count)
    print(f"Episode {episode}, Total Reward: {total_reward}")
    if episode % target_update_freq == 0:
        agent.update_target()

# Save the trained model weights
agent.q_net.cpu()  # Move to CPU before saving (optional)
torch.save(agent.q_net.state_dict(), "dqn_agent_weights.pth")
print("Model weights saved to dqn_agent_weights.pth")


# Trains the agent: tumorseg) daniel_kristensen@Daniels-MacBook-Air race-car % python -m src.game.agent.train_agent