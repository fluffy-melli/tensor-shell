from dataclasses import dataclass
import tensorflow as tf

@dataclass
class Flatten:
    shape: tuple

@dataclass
class Dense:
    units: int
    activation: str

@dataclass
class Dropout:
    rate: float

@dataclass
class MaxPooling2D:
    max: tuple

@dataclass
class Conv2D:
    units: int
    size: tuple
    activation: str
    shape: tuple

def Creaft(model_info: list) -> tf.keras.Model:
    model = tf.keras.models.Sequential()
    for info in model_info:
        if isinstance(info, Flatten):
            model.add(tf.keras.layers.Flatten(input_shape=info.shape))
        elif isinstance(info, Dense):
            model.add(tf.keras.layers.Dense(info.units, activation=info.activation))
        elif isinstance(info, Dropout):
            model.add(tf.keras.layers.Dropout(info.rate))
        elif isinstance(info, MaxPooling2D):
            model.add(tf.keras.layers.MaxPooling2D(info.max))
        elif isinstance(info, Conv2D):
            model.add(tf.keras.layers.Conv2D(info.units, info.size, activation=info.activation, input_shape=info.shape))
        else:
            raise ValueError(f"Unknown layer type: {type(info)}")
    return model