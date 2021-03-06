import numpy as np
from lib.torcs.gym_torcs_redwan import TorcsEnv
import argparse
import yaml
import os

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor


from dev.ac_v4 import model # select the right model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from lib.rewards import DefaultReward,ProgressiveSmoothingReward,HitReward
from lib.noises import ExplorationNoise

print('Gym Torcs Simulation Started ')

with open("param.yaml", 'r') as stream:
    param = yaml.load(stream)

state_size              =param['state_size']
nb_actions              =param['nb_actions']
seed                    =param['seed']

# Torcs parameters
ENV_NAME                =param['Torcs']['name']
vision                  =param['Torcs']['vision']
throttle                =param['Torcs']['throttle']
gear_change             =param['Torcs']['gear_change']
track                   =param['Torcs']['track']

# Training parameters
epochs                  =param['Training']['epochs']
loss                    =param['Training']['loss_metrics']
batch_size              =param['Training']['batch_size']
loss_metrics            =param['Training']['loss_metrics']
nb_steps_warmup_critic  =param['Training']['nb_steps_warmup_critic']
nb_steps_warmup_actor   =param['Training']['nb_steps_warmup_actor']
nb_max_episode_steps    =param['Training']['nb_max_episode_steps']

exp_dir                 =param['exp']
weight_save_interval    =param['Training']['weight_save_interval']
load_weights            =param['model']['load_weight']
alw                     =param['model']['actor_weight']
clw                     =param['model']['critic_weight']


class TorcsProcessor(Processor):
    total_reward = 0.
    _reward=DefaultReward()
    def process_observation(self, observation):
        ob = observation
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        self.obs ={'damage':ob.damage,'angle':ob.angle,'speedX':ob.speedX,'trackPos':ob.trackPos, 'distRaced':ob.distRaced}
        return s_t

    def process_state_batch(self, batch):
        batch = batch.reshape((-1,1,state_size))
        # print('batch ', np.shape(batch))
        return batch

    def process_reward(self,reward):
        reward=self._reward.reward(self.obs)
        self.total_reward+= reward
        # print('\t total_reward: {:.3f}'.format(self.total_reward))
        return reward

# ENV_NAME = 'Torcs-v0'

vision = False
env = TorcsEnv(vision=vision, throttle=throttle,gear_change=gear_change,track=track)
np.random.seed(seed)


def ensure_dir(file_path):
    file_path+='/checkpoints'
    directory = file_path
    if not os.path.exists(directory):
        print('making new dir at [{}]'.format(file_path))
        os.makedirs(directory,exist_ok=True)
    else:
        print('dir exists')

def save(agent):
    agent.save_weights('{}/ddpg_{}_final_weights.h5f'.format(exp_dir,ENV_NAME), overwrite=True)
    # agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=nb_max_episode_steps)

def get_agent(drlm):
    print('tesing', '.' * 60)
    actor = drlm.actor
    critic = drlm.critic
    nb_actions = drlm.nb_actions
    action_input = drlm.action_input
    processor = TorcsProcessor()

    # load weights

    print('loading weights ', load_weights, '.' * 60)

    if load_weights:
        actor.load_weights(alw)
        critic.load_weights(clw)

    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.5, mu=0., sigma=.3)

    # random_process = ExplorationNoise(nb_steps=epochs,
    #                                   epsilon=1.0,
    #                                   steer=OrnsteinUhlenbeckProcess(theta=0.6, mu=0, sigma=0.3),
    #                                   accel_brake=OrnsteinUhlenbeckProcess(theta=1.0, mu=0.5, sigma=0.3),
    #                                   noise=1)

    agent = DDPGAgent(nb_actions=nb_actions, batch_size=batch_size,
                      actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, processor=processor, nb_steps_warmup_critic=nb_steps_warmup_critic,
                      nb_steps_warmup_actor=nb_steps_warmup_actor,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.0001, clipnorm=1.), metrics=[loss])
    return agent

def train(drlm):
    agent = get_agent(drlm)
    # callbacks
    monitor = 'nb_episode_steps'
    mode = 'max'
    filename = exp_dir+'/checkpoints/weights.{epoch:02d}-{episode_reward:.2f}.hdf5'
    check_point = ModelCheckpoint(filepath=filename, monitor=monitor, verbose=1, save_best_only=False,
                                  mode=mode, period=weight_save_interval, save_weights_only=True)
    tensor_board = TensorBoard(log_dir='{}/graph/'.format(exp_dir))
    agent.fit(env, nb_steps=epochs, visualize=False, verbose=1, nb_max_episode_steps=nb_max_episode_steps,
              callbacks=[ tensor_board , check_point])

    return agent

def test(drlm):
    agent = get_agent(drlm)
    agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=nb_max_episode_steps)

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Torcs DRL')
    args.add_argument('--state_size', type=int, default=state_size)
    args.add_argument('--nb_actions',type=int, default=nb_actions)
    param = args.parse_args()
    ensure_dir(exp_dir)
    drlm = model(param)

    agent=train(drlm)
    save(agent)
    env.end()


