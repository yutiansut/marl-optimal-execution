import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Dropout
from tensorflow.keras.optimizers import RMSprop


class EvalModel(Model):
    def __init__(self, num_actions):
        super().__init__("mlp_q_network")
        self.layer1 = LSTM(64, return_sequences=True, recurrent_dropout=0.3)
        self.layer2 = LSTM(64, return_sequences=True, recurrent_dropout=0.3)
        self.layer3 = LSTM(64, recurrent_dropout=0.3)
        self.layer4 = Dense(20, activation="relu")
        self.layer5 = Dropout(0.3)
        self.layer6 = Dense(20, activation="relu")
        self.layer7 = Dropout(0.3)
        self.logits = Dense(num_actions, activation=None)

    def call(self, inputs):
        inputs = tf.reshape(inputs, [1, -1, 2])
        inputs = tf.cast(inputs, tf.float32)
        layer1 = self.layer1(inputs)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        logits = self.logits(layer7)
        return logits


class TargetModel(Model):
    def __init__(self, num_actions):
        super().__init__("mlp_q_network_1")
        self.layer1 = LSTM(64, return_sequences=True, recurrent_dropout=0.3)
        self.layer2 = LSTM(64, return_sequences=True, recurrent_dropout=0.3)
        self.layer3 = LSTM(64, recurrent_dropout=0.3)
        self.layer4 = Dense(20, activation="relu")
        self.layer5 = Dropout(0.3)
        self.layer6 = Dense(20, activation="relu")
        self.layer7 = Dropout(0.3)
        self.logits = Dense(num_actions, activation=None)

    def call(self, inputs):
        inputs = tf.reshape(inputs, [1, -1, 2])
        inputs = tf.cast(inputs, tf.float32)
        layer1 = self.layer1(inputs)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        logits = self.logits(layer7)
        return logits
