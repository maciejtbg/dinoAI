import time
from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from gym import Env
from gym.spaces import Box, Discrete

class WebGame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_location = {'top':150, 'left':130, 'width':250, 'height': 150}
        self.done_location = {'top':190, 'left':220, 'width':120, 'height':30}

    def step(self, action):
        action_map = {
            0:'space',
            1:'down',
            2:'no_op'
        }
        if action !=2:
            pydirectinput.press(action_map[action])

        done, done_cap = self.get_done()
        new_observation = self.get_observation()
        reward = 1
        info = {}
        return new_observation, reward, done, info
    
    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()


    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_observation()
    
    def close(self):
        cv2.destroyAllWindows()

    # def get_observation(self):
    #     raw = np.array(self.cap.grab(self.game_location))[:,:,3].astype(np.uint8)
    #     # gray=cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    #     resized = cv2.resize(raw, (100,83))
    #     channel = np.reshape(resized, (1,83  ,100))
    #     return channel  

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # resized= cv2.resize(gray, (150, 50))
        # channel = np.reshape(resized, (1,50,150))
        return gray
    
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

for episode in range(10): 
    print(f'Episode: {episode}.')
    obs = env.reset()
    done = False
    total_reward = 0 

    while not done:
        obs,reward, done, info = env.step(env.action_space.sample())
        total_reward +=reward
    print(f'Total Reward for episode {episode} is {total_reward}')

