from dataclasses import dataclass
import tensorflow as tf

@dataclass
class CompileD:
    optimizer: str
    loss: str
    metrics: list

def Compile(model: tf.keras.Model, info: CompileD):
    model.compile(
        optimizer=info.optimizer,
        loss=info.loss,
        metrics=info.metrics
    )

def Save(model: tf.keras.Model, filename: str):
    if not filename.endswith('.h5') and not filename.endswith('.keras'):
        filename += '.keras'
    model.save(filename)