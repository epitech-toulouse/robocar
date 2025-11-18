import sys
import os
import tensorflow as tf
from tensorflow.keras import layers, Model

def u_net(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    # Bridge
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(b)

    # Decoder
    u2 = layers.UpSampling2D(size=(2, 2))(b)
    u2 = layers.concatenate([u2, c2])
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)

    u1 = layers.UpSampling2D(size=(2, 2))(c3)
    u1 = layers.concatenate([u1, c1])
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u1)

    return Model(inputs, outputs)