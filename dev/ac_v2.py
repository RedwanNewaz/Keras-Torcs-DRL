from keras.models import Model
from keras.layers import Dense, Input, merge
from keras.layers import Dropout, BatchNormalization


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

    @property
    def actor(self):
        S = Input(shape=[self.state_size])
        h0 = Dense(self.HIDDEN1_UNITS, activation='relu')(S)
        h0 = BatchNormalization()(h0)
        h1 = Dense(self.HIDDEN2_UNITS, activation='relu')(h0)
        h1 = BatchNormalization()(h1)
        Steering = Dense(1,activation='tanh')(h1)
        Acceleration = Dense(1,activation='sigmoid')(h1)
        Brake = Dense(1,activation='sigmoid')(h1)
        V = merge([Steering,Acceleration,Brake],mode='concat')
        return  Model(input=S,output=V)

    @property
    def critic(self):
        S = Input(shape=[self.state_size])
        action_input = Input(shape=[self.nb_actions],name='action2')
        w1 = Dense(self.HIDDEN1_UNITS, activation='relu')(S)
        w1 = BatchNormalization()(w1)
        a1 = Dense(self.HIDDEN2_UNITS, activation='relu')(action_input)
        a1 = BatchNormalization()(a1)
        h1 = Dense(self.HIDDEN2_UNITS, activation='relu')(w1)
        h1 = BatchNormalization()(h1)
        h2 = merge([h1,a1],mode='sum')
        h3 = Dense(self.HIDDEN2_UNITS, activation='relu')(h2)
        h3 = BatchNormalization()(h3)
        V = Dense(1,activation='linear')(h3)

        self.action_input = action_input
        return Model(input=[S,action_input],output=V)