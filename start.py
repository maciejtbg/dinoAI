import time
from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from gymnasium import Env  # Zmienione na gymnasium
from gymnasium.spaces import Box, Discrete

class WebGame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_location = {'top':150, 'left':130, 'width':250, 'height': 150}
        self.done_location = {'top':190, 'left':220, 'width':120, 'height':30}

    # def step(self, action):
    #     action_map = {
    #         0:'space',
    #         1:'down',
    #         2:'no_op'
    #     }
    #     if action !=2:
    #         pydirectinput.press(action_map[action])

    #     done, done_cap = self.get_done()
    #     new_observation = self.get_observation()
    #     reward = 1
    #     info = {}
    #     return new_observation, reward, done, info

    def step(self, action):
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }
        if action != 2:
            pydirectinput.press(action_map[action])

        done, done_cap = self.get_done()
        new_observation = self.get_observation()
        reward = 1
        info = {}

        # Zakładamy, że w Twojej grze `done` oznacza zakończenie, czyli `terminated`.
        terminated = done  
        truncated = False  # Możesz ustawić `truncated` na `True`, jeśli masz warunki zakończenia związane z czasem.

        return new_observation, reward, terminated, truncated, info
    
    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            self.close()


    # def reset(self):
    #     time.sleep(1)
    #     pydirectinput.click(x=150, y=150)
    #     pydirectinput.press('space')
    #     return self.get_observation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Ustawienie seed, jeśli jest podane
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        observation = self.get_observation()
        info = {}  # Możesz dodać tu dodatkowe informacje, jeśli potrzebujesz
        return observation, info
    
    def close(self):
        cv2.destroyAllWindows()

    # def get_observation(self):
    #     raw = np.array(self.cap.grab(self.game_location))[:,:,3].astype(np.uint8)
    #     # gray=cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    #     resized = cv2.resize(raw, (100,83))
    #     channel = np.reshape(resized, (1,83  ,100))
    #     return channel  

    # def get_observation(self):
    #     raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
    #     gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    #     # resized= cv2.resize(gray, (150, 50))
    #     # channel = np.reshape(resized, (1,50,150))
    #     return gray
    
    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3]  # Pobranie obrazu o rozmiarze (150, 250)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)  # Konwersja na obraz w skali szarości
        resized = cv2.resize(gray, (100, 83))  # Skalowanie do wymiarów (100, 83)
        observation = np.expand_dims(resized, axis=0)  # Dodanie wymiaru kanału, aby uzyskać kształt (1, 83, 100)
        return observation


    def get_done(self):
        # done_cap = np.array(self.cap.grab(self.done_location))[:,:,3]
        done_cap = np.array(self.cap.grab(self.done_location))
        done_strings = ['GAME', 'GAHE']
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        print(res)
        if res in done_strings:
            done = True
        return done, done_cap
    
env = WebGame()
env.reset()
env.close()
env.render()
# plt.imshow(cv2.cvtColor(env.get_observation()[0], cv2.COLOR_BGR2RGB))
done, done_cap = env.get_done()
# plt.imshow(done_cap)



env= WebGame()
obs = env.get_observation()
# plt.imshow(cv2.cvtColor(obs[0], cv2.COLOR_BGR2RGB))
done,done_cap = env.get_done()
# plt.imshow(done_cap)
pytesseract.image_to_string(done_cap)[:4]



import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
env_checker.check_env(env)
# class TrainAndLoggingCallback(BaseCallback):
#     def __init__(self, check_freq, save_path, verbose=1):
#         super(TrainAndLoggingCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.save_path = save_path
    
#     def _init_callback(self):
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)
    
#     def _on_step(self):
#         if self.n_calls % self.check_freq == 0:
#             model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_class))
#             self.model_save(model_path)
#         return True
    
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

CHECKPOINT_DIR = "./train/"
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=300, save_path=CHECKPOINT_DIR)

from stable_baselines3 import DQN
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=100000, learning_starts=1000)

model.learn(total_timesteps=1000, callback=callback)


# for episode in range(1): 
#     print(f'Episode: {episode}.')
#     obs = env.reset()
#     done = False
#     total_reward = 0 

#     while not done:
#         action, _ =model.predict(obs)
#         obs, reward, done, info = env.step(int(action))
#         # obs,reward, done, info = env.step(env.action_space.sample())
#         total_reward +=reward
#     print(f'Total Reward for episode {episode} is {total_reward}')
#     time.sleep(2)