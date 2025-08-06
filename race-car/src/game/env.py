# src/game/env_gym.py
import gym
import numpy as np
from gym import spaces
from src.game import core
from src.game.core import GameState, initialize_game_state, get_observation_from_state, handle_action, update_cars, remove_passed_cars, place_car, intersects

class RaceCarEnv(gym.Env):
    def __init__(self, seed_value=None, sensor_removal=0):
        super().__init__()
        self.api_url = "http://example.com/api/predict"
        self.seed_value = seed_value
        self.sensor_removal = sensor_removal

        # Init game state in memory
        self._state = None
        self.max_ticks = core.MAX_TICKS

        # Action space: 5 diskrete handlinger
        self.action_space = spaces.Discrete(len(core.ACTION_LIST))

        # Observation space: 3 egenskaber for ego + 16 sensorer
        obs_dim = 3 + len(core.ACTION_LIST) * 0  # fallback hvis sensorer ikke er klar
        dummy_state = self._initialize_internal_state()
        obs_dim = len(get_observation_from_state(dummy_state))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _initialize_internal_state(self):
        # Brug core-funktion, men behold lokal kopi af state
        initialize_game_state(self.api_url, self.seed_value, self.sensor_removal)
        self._state = core.STATE
        return self._state

    def reset(self):
        self._initialize_internal_state()
        return np.array(get_observation_from_state(self._state), dtype=np.float32)

    def step(self, action_idx):
        action = core.ACTION_LIST[action_idx]
        handle_action(action)

        self._state.distance += self._state.ego.velocity.x
        update_cars()
        remove_passed_cars()
        place_car()

        for sensor in self._state.sensors:
            sensor.update()

        # Check for crash
        for car in self._state.cars:
            if car != self._state.ego and intersects(self._state.ego.rect, car.rect):
                self._state.crashed = True
        for wall in self._state.road.walls:
            if intersects(self._state.ego.rect, wall.rect):
                self._state.crashed = True

        obs = np.array(get_observation_from_state(self._state), dtype=np.float32)
        reward = self._compute_reward()
        done = self._state.crashed or self._state.ticks > self.max_ticks
        info = {"distance": self._state.distance}

        self._state.ticks += 1
        self._state.elapsed_game_time += 1000 // 60  # 60 FPS simuleret

        return obs, reward, done, info

    def _compute_reward(self):
        if self._state.crashed:
            return -100.0

        # Filtrér kun sensor-readings der IKKE er None
        valid_readings = [sensor.reading for sensor in self._state.sensors if sensor.reading is not None]
        #print("Valid readings:", valid_readings)
        # Hvis ingen valide readings, undgå min() og giv 0 straf
        if valid_readings:
            min_distance = min(valid_readings)
            danger_penalty = -1 if min_distance < 200 else 0
        else:
            danger_penalty = 0

        step_penalty = 0.0
        #distance_reward = self._state.ego.velocity.x
        distance_reward = 0 #Nul for at udforske om den kan undvige biler

        return distance_reward + danger_penalty + step_penalty


    def render(self, mode="human"):
        # Optional – kun hvis du vil bruge visning
        pass

    def close(self):
        pass
