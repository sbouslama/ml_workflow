# entity_emebddings.py 

import os 
import gc 
import joblib 
import pandas as pd
import numpy as np 
from sklearn import metrics, preprocessing 
from tensorflow.keras import layers
from tensorflow.keras import optimizers 
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras import callbacks 
from tensorflow.keras import backend as K 
from tensorflow.keras import utils

def create_model(data, catcols):     
    """     
    This function returns a compiled tf.keras model for entity embeddings     
    :param data: this is a pandas dataframe     
    :param catcols: list of categorical column names     
    
    :return: compiled tf.keras model    
    """     
    # init list of inputs for embeddings     
    
    inputs = [] 
 
    # init list of outputs for embeddings     
    
    outputs = [] 
 
    # loop over all categorical columns     
    for c in catcols:         
        # find the number of unique values in the column         num_unique_values = int(data[c].nunique())         
        # simple dimension of embedding calculator        
        # min size is half of the number of unique values         
        # max size is 50. max size depends on the number of unique        
        # categories too. 50 is quite sufficient most of the times         
        # but if you have millions of unique values, you might need         
        # a larger dimension         
        
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50)) 
 
        # simple keras input layer with size 1         
        inp = layers.Input(shape=(1,)) 

        # add embedding layer to raw input         
        # embedding size is always 1 more than unique values in input           
        out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)

        # 1-d spatial dropout is the standard for emebedding layers          
        # you can use it in NLP tasks too         
        out = layers.SpatialDropout1D(0.3)(out) 
 
        # reshape the input to the dimension of embedding
        # this becomes our output layer for current feature         
        out = layers.Reshape(target_shape=(embed_dim, ))(out) 
 
        # add input to input list        
        inputs.append(inp) 
 
        # add output to output list         
        outputs.append(out) 

    # concatenate all output layers     
    x = layers.Concatenate()(outputs) 
 
    # add a batchnorm layer.     
    # from here, everything is up to you    
    # you can try different architectures     
    # this is the architecture I like to use    
    # if you have numerical features, you should add    
    # them here or in concatenate layer     
    x = layers.BatchNormalization()(x)          
    # a bunch of dense layers with dropout.     
    # start with 1 or two layers only     
    
    x = layers.Dense(300, activation="relu")(x)     
    x = layers.Dropout(0.3)(x)     
    x = layers.BatchNormalization()(x)          
    x = layers.Dense(300, activation="relu")(x)     
    x = layers.Dropout(0.3)(x)     
    x = layers.BatchNormalization()(x)          
    # using softmax and treating it as a two class problem    
    # you can also use sigmoid, then you need to use only one     
    # output class     
    y = layers.Dense(2, activation="softmax")(x) 
 
    # create final model     
    model = Model(inputs=inputs, outputs=y) 
 
    # compile the model     
    # we use adam and binary cross entropy.     
    # feel free to use something else and see how model behaves     
    model.compile(loss='binary_crossentropy', optimizer='adam')     
    return model