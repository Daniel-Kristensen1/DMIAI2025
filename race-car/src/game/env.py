from src.game import core
import numpy as np

print("In env.py: id(STATE):", id(core.STATE))
ACTION_LIST = ["ACCELERATE", "DECELERATE", "STEER_LEFT", "STEER_RIGHT", "NOTHING"]

class RaceCarEnv:
    def __init__(self):
        self.api_url = "http://example.com/api/predict"
        self.seed_value = None

    def reset(self):
        core.initialize_game_state(self.api_url, self.seed_value)
        return self.get_observation()

    def step(self, action_idx):
        action = ACTION_LIST[action_idx]
        core.update_game(action)  # Uses global STATE
        obs = self.get_observation()
        reward = self.compute_reward()
        done = core.STATE.crashed or core.STATE.ticks > core.MAX_TICKS
        info = {}
        return obs, reward, done, info

    def get_observation(self):
        obs = [core.STATE.ego.x, core.STATE.ego.y, core.STATE.ego.velocity.x]
        obs += [s.reading if s.reading is not None else -1.0 for s in core.STATE.sensors]
        return np.array(obs, dtype=np.float32)


    def compute_reward(self):
        if core.STATE.crashed:
            return -100
        return core.STATE.ego.velocity.x  # Or use distance increment