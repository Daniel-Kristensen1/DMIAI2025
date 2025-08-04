import pygame
from src.game.core import initialize_game_state, game_loop
from src.game.agent.dqn_agent import DQNAgent
import torch

pygame.init()
initialize_game_state("http://example.com/api/predict", None)

# Set up agent
obs_dim = ...  # set to your observation size
action_dim = 5
agent = DQNAgent(obs_dim, action_dim)
agent.q_net.load_state_dict(torch.load("dqn_agent_weights.pth"))
agent.q_net.eval()
agent.epsilon = 0.0

game_loop(verbose=True, agent=agent)
pygame.quit()