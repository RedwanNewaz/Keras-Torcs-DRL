from keras.models import Model
from keras.layers import Dense, Input, merge
from keras.layers import Dropout, BatchNormalization
from keras import backend as K


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

    def proximal_policy_optimization_loss(actual_value, predicted_value, old_prediction):
        advantage = actual_value - predicted_value

        def loss(y_true, y_pred):
            prob = K.sum(y_true * y_pred)
            old_prob = K.sum(y_true * old_prediction)
            r = prob / (old_prob + 1e-10)

            return -K.log(prob + 1e-10) * K.mean(
                K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage))

        return loss

    @property
    def actor(self):
        actual_value = Input(shape=(1,))
        predicted_value = Input(shape=(1,))
        old_prediction = Input(shape=(self.nb_actions,))
        state_input = Input(shape=[self.state_size])
        h0 = Dense(self.HIDDEN1_UNITS, activation='relu')(state_input)
        h0 = BatchNormalization()(h0)
        h1 = Dense(self.HIDDEN2_UNITS, activation='relu')(h0)
        h1 = BatchNormalization()(h1)
        Steering = Dense(1,activation='tanh')(h1)
        Acceleration = Dense(1,activation='sigmoid')(h1)
        Brake = Dense(1,activation='sigmoid')(h1)
        out_action = merge([Steering,Acceleration,Brake],mode='concat')

        self.ppo_loss = self.proximal_policy_optimization_loss(
                          actual_value=actual_value,
                          old_prediction=old_prediction,
                          predicted_value=predicted_value)
        return  Model(input=[state_input, actual_value, predicted_value, old_prediction],output=[out_action])

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