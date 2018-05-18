from keras.models import Model,Sequential
from keras.layers import Dense, Input, merge
from keras.layers import Flatten
from keras.initializers import VarianceScaling
from keras.activations import softmax
import argparse


class model(object):
    def __init__(self, arg):
        self.arg = arg
        self.nb_actions = arg.nb_actions
        self.state_size = arg.state_size



    @property
    def actor(self):
        observation_input = Input(shape=(1,) + (self.state_size,),name='observation_input')
        h0 = Dense(200, activation='relu', init='he_normal')(Flatten()(observation_input))
        h1 = Dense(200, activation='relu', init='he_normal')(h0)
        output = Dense(self.nb_actions, activation='tanh', init='he_normal')(h1)
        model = Model(input=observation_input, output=output)
        return  model

    @property
    def critic(self):
        action_input = Input(shape=(self.nb_actions,))
        observation_input = Input(shape=(1,) + (self.state_size,),name='observation_input')
        w1 = Dense(200, activation='relu', init='he_normal')(Flatten()(observation_input))
        a1 = Dense(200, activation='linear', init='he_normal')(action_input)
        h1 = Dense(200, activation='linear', init='he_normal')(w1)
        h2 = merge([h1, a1], mode='concat')
        h3 = Dense(100, activation='relu', init='he_normal')(h2)
        output = Dense(1, activation='linear', init='he_normal')(h3)


        self.action_input = action_input
        model = Model(input=[action_input, observation_input], output=output)
        return model

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Torcs DRL')
    args.add_argument('--state_size', type=int, default=29)
    args.add_argument('--nb_actions',type=int, default=3)
    param = args.parse_args()
    m =model(param)
    c=m.critic
    c.summary()
