import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop


class NNModel_1(Model):
    def __init__(self, num_actions):
        super().__init__("mlp_q_network")
        self.layer1 = Dense(32, activation="relu")
        self.layer2 = Dense(64, activation="relu")
        self.layer3 = Dense(128, activation="relu")
        self.layer4 = Dense(128, activation="relu")
        self.layer5 = Dense(64, activation="relu")
        self.layer6 = Dense(32, activation="relu")
        self.dropout = Dropout(0.1)
        self.logits = Dense(num_actions, activation=None)

    def call(self, inputs):
        layer1 = self.layer1(inputs)
        layer2 = self.dropout(self.layer2(layer1))
        layer3 = self.dropout(self.layer3(layer2))
        layer4 = self.dropout(self.layer4(layer3))
        layer5 = self.dropout(self.layer5(layer4))
        layer6 = self.dropout(self.layer6(layer5))
        logits = self.logits(layer6)
        return logits


class NNModel_2(Model):
    def __init__(self, num_actions):
        super().__init__("mlp_q_network")
        self.layer1 = Dense(32, activation="relu")
        self.layer2 = Dense(64, activation="relu")
        self.layer3 = Dense(128, activation="relu")
        self.layer4 = Dense(256, activation="relu")
        self.layer5 = Dense(128, activation="relu")
        self.layer6 = Dense(64, activation="relu")
        self.layer7 = Dense(32, activation="relu")
        self.dropout = Dropout(0.1)
        self.logits = Dense(num_actions, activation=None)

    def call(self, inputs):
        layer1 = self.layer1(inputs)
        layer2 = self.dropout(self.layer2(layer1))
        layer3 = self.dropout(self.layer3(layer2))
        layer4 = self.dropout(self.layer4(layer3))
        layer5 = self.dropout(self.layer5(layer4))
        layer6 = self.dropout(self.layer6(layer5))
        layer7 = self.dropout(self.layer7(layer6))
        logits = self.logits(layer7)
        return logits


class EvalModel(NNModel_1):
    pass


class TargetModel(NNModel_1):
    pass
