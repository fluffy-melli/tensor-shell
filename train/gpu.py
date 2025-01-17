import tensorflow as tf
def Device():
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        print("\033[42mGPU detected: Using GPU\033[0m")
        tf.config.set_visible_devices(gpus[0], 'GPU')
        return 'GPU'
    else:
        print("\033[41mNo GPU detected: Using CPU\033[0m")
        tf.config.set_visible_devices([], 'GPU')
        return 'CPU'