import numpy as np
import os
import os.path
import tensorflow as tf

np.random.seed(400)

# Import keras main libraries
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Input,Bidirectional,Layer
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K


mini_batch_size, num_epochs = 128,30 
input_size = 252
number_units = 256
number_layers = 3
number_classes = 88
best_accuracy = 0
size_samples = 100

#Arg inputs
data_directory = './mat_multiple/'
weights_dir = './weights/'

class attention_multi(Layer):
    def __init__(self, return_sequences=True, heads = 3, head_size = 64):
        self.heads = heads
        self.head_size = head_size
        self.return_sequences = return_sequences
        super(attention_multi,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],self.heads*self.head_size)
                               ,initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],self.heads*self.head_size)
                               ,initializer="normal")
        self.o=self.add_weight(name="out_weight", shape=(self.head_size,input_shape[-1])
                               ,initializer="normal")
        self.o_b=self.add_weight(name="out_bias", shape=(input_shape[1],1)
                               ,initializer="normal")
        super(attention_multi,self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        e = tf.reshape(e,shape = (tf.shape(e)[0],self.heads,tf.shape(e)[1],self.head_size))
        a = K.sum(e,axis=1)
        a = K.tanh(K.dot(a,self.o) + self.o_b)
        a = K.softmax(a)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)

def load_valid():
    X_val_full = []
    y_val_full = []
    num_tr_batches = len([name for name in os.listdir(data_directory + "val/")])/2
    print ('Load validation data...')
    for i in range(int(num_tr_batches)):
        print ("Batching..." + str(i) + "val_X.npy")
        X_val = np.load(data_directory + "val/" + str(i) + "val_X.npy" )
        max_shape = (X_val.shape[0]//100)*100
        X_val = np.array(np.reshape(X_val[0:max_shape,:],(X_val.shape[0]//size_samples,size_samples,input_size)))

        print ("Batching..." + str(i) + "val_y.npy")
        y_val = np.load(data_directory + "val/" + str(i) + "val_y.npy" )
        y_val = np.array(np.reshape(y_val[0:max_shape,:],(y_val.shape[0]//size_samples,size_samples,number_classes)))
        if i == 0:
            X_val_full = X_val
            y_val_full = y_val
        else:
            X_val_full = np.concatenate((X_val_full,X_val), axis = 0)
            y_val_full = np.concatenate((y_val_full,y_val), axis = 0)
    with tf.device('/cpu:0'):
        X_val_tensor = tf.convert_to_tensor(X_val_full, np.float32)
        del X_val_full
        y_val_tensor = tf.convert_to_tensor(y_val_full, np.uint8)
        del y_val_full

    return X_val_tensor, y_val_tensor

def load_train():
    X = []
    y = []
    num_tr_batches = len([name for name in os.listdir(data_directory + "train/")])/2
    print ('Loading all data')
    for i in range(int(num_tr_batches)):
        print ("Batching..." + str(i) + "train_X.npy")
        X_train = np.load(data_directory + "train/" + str(i) + "train_X.npy" )
        max_shape = (X_train.shape[0]//100)*100 
        X_train = np.array(np.reshape(X_train[0:max_shape,:],(X_train.shape[0]//size_samples,size_samples,input_size)))

        print ("Batching..." + str(i) + "train_y.npy")
        y_train = np.load(data_directory + "train/" + str(i) + "train_y.npy" )
        y_train = np.array(np.reshape(y_train[0:max_shape,:],(y_train.shape[0]//size_samples,size_samples,number_classes)))
        if i == 0:
            X = X_train
            y = y_train
        else:
            X = np.concatenate((X,X_train), axis = 0)
            y = np.concatenate((y,y_train), axis = 0)
    with tf.device('/cpu:0'):
        X_train_tensor = tf.convert_to_tensor(X, np.float32)
        del X
        y_train_tensor = tf.convert_to_tensor(y, np.uint8)
        del y

    return X_train_tensor, y_train_tensor


def custom_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):
        y_true = K.cast(y_true, dtype=tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Compute cross entropy from probabilities.
        bce = y_true * tf.math.log(y_pred + epsilon)
        bce += (1 - y_true) * tf.math.log(1 - y_pred + epsilon)
        bce = -bce

        # Apply the weights to each class individually
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_bce = weight_vector * bce

        # Return the mean error
        return tf.reduce_mean(weighted_bce)
    return weighted_binary_crossentropy

def build_model(hidden_units, classes=number_classes, input_shape=(size_samples,input_size), dropout=0.2):
        model = Sequential()

        ## --------------- ENCODER ----------------##
        model.add(Input(shape=(input_shape)))
        model.add(attention_multi(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(hidden_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(dropout))
        ## --------------- DECODER ----------------##
        # model.add(attention(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(hidden_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(dropout))
        # model.add(attention(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(hidden_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(dropout))
        model.add(Dense(hidden_units, activation="tanh"))
        model.add(Dense(classes, activation='sigmoid'))
        model.compile() 
        model.build(input_shape=(size_samples, input_size))
        model.compile(loss=custom_weighted_binary_crossentropy(0.65,0.35), optimizer='adam', metrics=['binary_accuracy'])
        model.summary()
        return model


def train_model():
    # Load data
    X_val_tensor, y_val_tensor = load_valid()
    X_train_tensor, y_train_tensor = load_train()

    # Build model
    model = build_model(hidden_units=number_units,  classes=number_classes, input_shape=(size_samples,input_size), dropout=0.2)

    checkpointer = ModelCheckpoint(filepath= weights_dir + "MultiheadAttention_BiLSTM.hdf5", verbose=1, save_best_only=True, save_weights_only = True)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

    training_log = open(weights_dir + "Training.log", "w")
    print ('Train . . .')
    save = model.fit(X_train_tensor, y_train_tensor,epochs = 100,verbose=1,validation_data=(X_val_tensor, y_val_tensor), callbacks=[checkpointer,early])
    training_log.write(str(save.history) + "\n")
    training_log.close()

    return model

if  __name__ == '__main__':
    model = train_model()