# model_lstm.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

def build_temporal_model(input_shape=(16,224,224,3), n_classes=5, lstm_units=256, dropout=0.4, train_base=False):

    seq_len = input_shape[0]

    base_cnn = EfficientNetB0(include_top=False, weights='imagenet', pooling=None, input_shape=input_shape[1:])
    base_cnn.trainable = train_base  # set trainability externally as well

    seq_input = layers.Input(shape=input_shape, name='seq_input')  # batch x seq x H x W x C

    td = layers.TimeDistributed(base_cnn, name='td_cnn')(seq_input)  # (b, seq, h', w', c')

    td = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name='td_gap')(td)  # (b, seq, feat)

    td = layers.TimeDistributed(layers.Dense(512, activation='relu'), name='td_proj')(td)  # (b, seq, 512)

    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True), name='bilstm')(td)  # (b, seq, 2*lstm)
    x = layers.TimeDistributed(layers.Dense(256, activation='relu'), name='td_fc')(x)
    x = layers.TimeDistributed(layers.Dropout(dropout), name='td_dropout')(x)
    out = layers.TimeDistributed(layers.Dense(n_classes, activation='softmax'), name='td_out')(x)  # (b, seq, n_classes)

    model = Model(inputs=seq_input, outputs=out)
    return model
