import torch
from src.game.env import RaceCarEnv
from src.game.agent.dqn_agent import DQNAgent
from src.game import core
import pygame

def main():
    pygame.init()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = RaceCarEnv()
    obs = env.reset()
    print(f"Observation from env.reset(): {obs}")
    obs_dim = len(obs)
    action_dim = 5

    agent = DQNAgent(obs_dim, action_dim)
    agent.q_net.load_state_dict(torch.load("dqn_agent_weights_parallel.pth", map_location=device))
    agent.q_net.to(device)
    agent.q_net.eval()  # Sæt i eval mode (vigtigt!)
    print("Using device:", device)

    # Initialiser spil state, seed hvis du vil
    core.initialize_game_state(env.api_url, seed_value=None)

    # Start pygame med agent i game loop
    core.game_loop(verbose=True, log_actions=False, agent=agent)

if __name__ == "__main__":
    main()
