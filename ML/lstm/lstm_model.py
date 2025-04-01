import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Layer, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Wa = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        et = K.tanh(K.dot(inputs, self.Wa) + self.b)
        at = K.softmax(K.dot(et, self.u), axis=1)
        output = K.sum(inputs * at, axis=1)
        return output

def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(units=46, return_sequences=True)(inputs)
    dropout_out = Dropout(0.26)(lstm_out)
    attention_out = Attention()(dropout_out) 
    output = Dense(1, activation='linear')(attention_out) 

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    return model