"""
Convolutional architectures with embedded Log Mel Spectrogram computation
"""
import tensorflow as tf
import tensorflow.keras as keras
from LogMelSpec import LogMelSpec

#from . import efficientnet


def calc_Logmel(*,
                   samples: int, classes: int, dropout_rate: float = 0.2,
                   mel_params: dict, net_params: dict = None
                   ):
      # Input tensor, audio samples
    input_tensor = keras.Input(shape=(samples,))

    # Log Mel Spectrogram on the fly
    log_mel_spec_layer = LogMelSpec(**mel_params)
    x = log_mel_spec_layer(input_tensor)
    model = keras.Model(input_tensor, x, name=f'logmel')
    return model

def simplenet2att2(*,
                   samples: int, classes: int, dropout_rate: float = 0.2,
                   mel_params: dict, net_params: dict = None
                   ):
    """
    Get a Simple net model with embedded Log Mel Spectrogram computation and attention
    Args:
        samples:
        classes:
        dropout_rate:
        mel_params:
        net_params:

    Returns:

    """

    # Input tensor, audio samples
    #input_tensor = keras.Input(shape=(samples[:],))
    input_tensor = keras.Input(shape=(76,61))


    # Log Mel Spectrogram on the fly
    #log_mel_spec_layer = LogMelSpec(**mel_params)
    #x = log_mel_spec_layer(input_tensor)
    #print(x.shape)
    x=input_tensor    


    # Learn data scaling as batch normalization on frequencies only
    x = keras.layers.BatchNormalization(moving_mean_initializer=tf.constant_initializer(-12),
                                        moving_variance_initializer=tf.constant_initializer(288),
                                        name='logmelnorm')(x)

    # Expand as 2D CNN. X has shape (N,T,F,1) (N=Batch size, T=Time, F=Frequency)
    x = tf.keras.layers.Reshape(tuple(x.shape[1:3]) + (1,), name='reshape')(x)

    # Simple net
    x = keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, activation='relu', name='conv1')(x)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', name='conv2')(x)

    # Attention
    att = keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', name='attconv')(x)
    att = tf.reduce_mean(att, axis=1, keepdims=True)  # Averaging over time. Attention is a F/4 elements vector
    x = keras.layers.Multiply(name='attmult')([att, x])

    x = keras.layers.GlobalAveragePooling2D(name='global_pool')(x)

    # Final dropout and predictions
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
    x = keras.layers.Dense(
        classes,
        activation='softmax',
        kernel_initializer={
            'class_name': 'VarianceScaling',
            'config': {
                'scale': 1. / 3.,
                'mode': 'fan_out',
                'distribution': 'uniform'
            }
        },
        name='predictions')(x)

    # Create the final model
    inputs = keras.utils.get_source_inputs(input_tensor)
    model = keras.Model(inputs, x, name=f'simplenet2att2logmel')

    return model
