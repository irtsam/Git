"""
Example script to run a Sound Event Detection Model, from data loading to post-processing
"""
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf

assert float(tf.__version__[:3]) >= 2.3
import json
from pathlib import Path
import numpy as np
import librosa
from tensorflow import keras




''' The version to run the complete quantization script must
be higher than 2.3'''
assert float(tf.__version__[:3]) >=2.3







class QuantModel():

    def __init__(self, model=tf.keras.Model,data=[]):
        '''
        1. Accepts a keras model, long term will allow saved model and other formats
        2. Accepts a numpy or tensor data of the format such that indexing such as
        data[0] will return one input in the correct format to be fed forward through the
        network
        '''
        self.data=data
        self.model=model

    
    '''Added script to quantize model and allows custom ops
    for Logmelspectrogram operations (Might cause mix quantization)'''
    def quant_model_int8(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.representative_dataset=self.representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        #converter.allow_custom_ops=True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model_quant = converter.convert()
        open("converted_model2.tflite",'wb').write(tflite_model_quant)
        return tflite_model_quant




    '''Returns a tflite model with no quantization i.e. weights and variable data all
    in float32'''
    def convert_tflite_no_quant(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        open("converted_model.tflite",'wb').write(tflite_model)
        return tflite_model


    def representative_data_gen(self):
        # Model has only one input so each data point has one element.
        yield [self.data]
