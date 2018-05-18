from gym import spaces
import numpy as np
import copy
import collections as col
import os
import time
from .redwan_snakeoil3_gym import Client


class TorcsEnv:
    terminal_judge_start = 1000  # If after 100 timestep still no progress, terminated
    termination_limit_progress = 0  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False, track='practice.xml'):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.initial_run = True
        self.__time_stop = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.observation_space = spaces.Box(low=0, high=0, shape=(29,))

        self.client = Client(p=3001, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

    def step(self, u):
        client = self.client
        this_action = self.agent_to_torcs(u)
        action_torcs = client.R.d

        # __decode_action_data
        action_torcs['steer'] = this_action['steer']  # Steering in [-1, 1]
        action_torcs =self._get_throttle(action_torcs, this_action) #  Simple Autnmatic Throttle Control by Snakeoil
        action_torcs = self._get_gear(action_torcs, this_action)#  Automatic Gear Change by Snakeoil

        obs,obs_pre,self.observation = self._get_observation()
        # Reward setting Here #######################################
        # direction-dependent positive reward
        reward, progress = self._get_reward(obs,obs_pre)
        # done = self._get_done(obs, progress) TODO: never use it since reward is different in porcessor (main.py>>processor)
        done = self.__check_done(obs)

        self.time_step += 1

        return self.observation, reward, done , {}

    def reset(self, relaunch=False):
        print("Reset")
        self._reset
        self.time_step = 0

        if self.initial_reset is not True:
            # self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                # self.reset_torcs()
                try:
                    self.client.restart
                    print("### TORCS is RELAUNCHED ###")
                except:
                    print('cannot relaunch')
                    return


        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.observation

    @property
    def _reset(self):
        self.client = Client(p=3001, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

    def end(self):
        os.system('pkill torcs')




    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        return np.array([r, g, b], dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel', 'curLapTime', 'distFromStart', 'distRaced']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32), curLapTime=np.array(raw_obs['curLapTime'], dtype=np.float32)/1.0, distFromStart = np.array(raw_obs['distFromStart'], dtype=np.float32)/1.0, distRaced=np.array(raw_obs['distRaced'], dtype=np.float32)/1.0 )
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
##########################################################################

    def _get_throttle(self,  action_torcs,this_action):
        client = self.client
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer'] * 50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1 / (client.S.d['speedX'] + .1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2] + client.S.d['wheelSpinVel'][3]) -
                    (client.S.d['wheelSpinVel'][0] + client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        return action_torcs

    def _get_gear(self,action_torcs,this_action):
        client = self.client
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        return action_torcs

    def _get_observation(self):
        client = self.client
        # Get observation  #################################
        # Apply the Agent's action into torcs
        obs_pre = copy.deepcopy(client.S.d)  # Save the privious full-obs from torcs for the reward calculation
        client.respond_to_server()
        client.get_servers_input()
        obs = client.S.d
        return obs,obs_pre,self.make_observaton(obs)

    def _get_reward(self,obs,obs_pre):
        sp = np.array(obs['speedX'])
        # damage = np.array(obs['damage'])
        # rpm = np.array(obs['rpm'])
        # track = np.array(obs['track'])
        # trackPos = np.array(obs['trackPos'])

        progress = sp * np.cos(obs['angle']) - np.abs(sp * np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1
        return reward, progress

    def _get_done(self, obs,progress):
        '''
        :param obs: measurement
        :param progress: reward
        :return: boolean done
        '''
        client = self.client
        episode_terminate = False
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        if self.terminal_judge_start < self.time_step:
            if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
                print('This One', track.any(), trackPos)
                reward = -200
                episode_terminate = True
                client.R.d['meta'] = True

        elif self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                print('This Two', progress, sp, obs['distRaced'])
                print("No progress")
                episode_terminate = True
                client.R.d['meta'] = True

        elif np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            print('This Three')
            episode_terminate = True
            client.R.d['meta'] = True


        elif client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.shutdown()
            # client.respond_to_server()

        return episode_terminate

    def __check_done(self, sensors):
        if sensors['speedX'] < self.termination_limit_progress:
            self.__time_stop += 1
        else:
            self.__time_stop = 0
        return self.__time_stop > self.terminal_judge_start or np.abs(sensors['trackPos']) > 0.99 \
               or sensors['damage'] > 0
