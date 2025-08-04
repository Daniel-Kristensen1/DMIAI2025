
# Trains the agent: tumorseg) daniel_kristensen@Daniels-MacBook-Air race-car % cond.agent.train_agent
from src.game.env import RaceCarEnv
from src.game.agent.dqn_agent import DQNAgent
from tqdm import trange
import torch


torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())  # Skal nu vise True hvis alt er i orden
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

env = RaceCarEnv()
obs_dim = len(env.reset())
action_dim = 5

agent = DQNAgent(obs_dim, action_dim)  
agent.q_net.to(device)
agent.target_net.to(device)  # Husk også target net på GPU

num_episodes = 100
target_update_freq = 10

for episode in trange(num_episodes, desc="Training Progress"):
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    done = False
    total_reward = 0
    max_steps = 1000
    step_count = 0

    while not done and step_count < max_steps:
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)

        # Her gemmer vi CPU numpy arrays i memory for at undgå GPU-memory leaks
        agent.store_transition(
            obs.cpu().numpy(),  # konverter til numpy på CPU
            action,
            reward,
            next_obs.cpu().numpy(),  # konverter til numpy på CPU
            done
        )

        agent.learn()

        obs = next_obs
        total_reward += reward
        step_count += 1

    print(f"Episode {episode} finished after {step_count} steps, total reward: {total_reward}")

    if episode % target_update_freq == 0:
        agent.update_target()

# Gem modellen på CPU for kompatibilitet
agent.q_net.cpu()
torch.save(agent.q_net.state_dict(), "dqn_agent_weights.pth")
print("Model weights saved to dqn_agent_weights.pth")
