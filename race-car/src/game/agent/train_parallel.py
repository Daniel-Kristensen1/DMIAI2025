import sys
sys.path.append("src/game")


from src.game.env import RaceCarEnv
import torch
import numpy as np
from src.game.agent.dqn_agent import DQNAgent



from tqdm import trange

episodes = 2000

def worker(worker_id, num_episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = RaceCarEnv()
    obs_dim = len(env.reset())
    action_dim = 5
    agent = DQNAgent(obs_dim, action_dim)
    agent.q_net.to(device)
    agent.target_net.to(device)

    target_update_freq = 10
    experiences = []  # <-- new list to store transitions

    for episode in trange(num_episodes, desc=f"Worker {worker_id} Progress"):
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

            agent.store_transition(
                obs.cpu().numpy(),
                action,
                reward,
                next_obs.cpu().numpy(),
                done
            )

            # Save the experience for return
            experiences.append((
                obs.cpu().numpy(),
                action,
                reward,
                next_obs.cpu().numpy(),
                done
            ))

            agent.learn()

            obs = next_obs
            total_reward += reward
            step_count += 1

        if episode % target_update_freq == 0:
            agent.update_target()

        print(f"Worker {worker_id} - Episode {episode} finished after {step_count} steps, total reward: {total_reward}")
        if 'distance' in info and info['distance'] is not None:
            print(f"Distance: {info['distance']}")
        else:
            print("Distance info not available.")

    torch.save(agent.q_net.cpu().state_dict(), f"dqn_agent_worker_{worker_id}_weights.pth")
    print(f"Worker {worker_id} model weights saved.")

    return experiences  # <-- return all experiences collected
import multiprocessing as mp

def parallel_training(total_episodes=episodes, workers=4):
    episodes_per_worker = total_episodes // workers

    with mp.Pool(workers) as pool:
        results = pool.starmap(worker, [(i, episodes_per_worker) for i in range(workers)])

    print("Alle workers er færdige.")

    # Saml oplevelser fra alle workers
    all_experiences = []
    for worker_exp in results:
        all_experiences.extend(worker_exp)

    print(f"Total experiences collected: {len(all_experiences)}")

    # Nu kan du træne agenten på disse oplevelser i hovedprocessen
    # F.eks.:
    env = RaceCarEnv()
    obs_dim = len(env.reset())
    action_dim = 5
    agent = DQNAgent(obs_dim, action_dim)

    for obs, action, reward, next_obs, done in all_experiences:
        agent.store_transition(obs, action, reward, next_obs, done)

    for _ in range(len(all_experiences) // agent.batch_size * 5):
        agent.learn()


    agent.update_target()

    # Gem model
    torch.save(agent.q_net.state_dict(), "dqn_agent_weights_parallel.pth")
    print("Træning færdig og model gemt.")

if __name__ == "__main__":
    parallel_training()


    # Evaluer den trænede model
    from src.game.env import RaceCarEnv
    from src.game.agent.dqn_agent import DQNAgent
    import torch

    # Init environment
    env = RaceCarEnv()

    # Init agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(obs_dim, action_dim)

    # Load model weights
    agent.q_net.load_state_dict(torch.load("dqn_agent_weights_parallel.pth"))
    agent.q_net.eval()

    # Eval loop
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}, Distance: {info['distance']}")