from keras.models import Model,Sequential
from keras.layers import Dense, Input, merge
from keras.layers import Dropout, BatchNormalization, Activation
from keras.initializers import VarianceScaling
from keras.activations import softmax
import argparse


'''
The basic models have been modified here. We understand the importance of having larger hidden dimensions,
so kept these unchanged. However, in critic model the output layer has been modified from 3 to 1.
Instead of giving criticism to 3 individual actions, we opt to criticize the overall performance.
Therefore, the activation functions of the critics layers have been modified accordingly. 
Here we will introduce the Batch normalization and also increase the hidden layers
'''


class model(object):
    HIDDEN1_UNITS = 150*2
    HIDDEN2_UNITS =  300*2
    def __init__(self, arg):
        self.arg = arg
        self.nb_actions = arg.nb_actions
        self.state_size = arg.state_size
        self.rint = VarianceScaling(scale=1e-4, mode='fan_in', distribution='normal', seed=None)

    @property
    def actor(self):
        S = Input(shape=[self.state_size])
        h0 = Dense(self.HIDDEN1_UNITS, activation='relu', init='uniform')(S)
        # h0 = BatchNormalization()(h0)
        h1 = Dense(self.HIDDEN2_UNITS, activation='relu')(h0)
        # h1 = BatchNormalization()(h1)
        Steering = Dense(1,activation='tanh', init=self.rint)(h1)
        Acceleration = Dense(1,activation='sigmoid', init=self.rint)(h1)
        Brake = Dense(1,activation='sigmoid', init=self.rint)(h1)
        V = merge([Steering,Acceleration,Brake],mode='concat')
        model = Model(input=S, output=V)





        return  model

    @property
    def critic(self):
        S = Input(shape=[self.state_size])
        action_input = Input(shape=[self.nb_actions],name='action2')
        w1 = Dense(self.HIDDEN1_UNITS, activation='relu', init='uniform')(S)

        # w1 = BatchNormalization()(w1)

        a1 = Dense(self.HIDDEN2_UNITS, activation='relu', init='uniform')(action_input)
        # a1 = BatchNormalization()(a1)

        h1 = Dense(self.HIDDEN2_UNITS, activation='relu')(w1)
        # h1 = BatchNormalization()(h1)

        h2 = merge([h1,a1],mode='sum')

        h3 = Dense(self.HIDDEN1_UNITS, activation='relu')(h2)
        # h3 = BatchNormalization()(h3)
        V = Dense(1,activation='linear')(h3)


        self.action_input = action_input
        model = Model(input=[S,action_input],output=V)
        return model

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Torcs DRL')
    args.add_argument('--state_size', type=int, default=29)
    args.add_argument('--nb_actions',type=int, default=3)
    param = args.parse_args()
    m =model(param)
    c=m.critic
    c.summary()
