import base64
import os

import keras.backend as K
import librosa
import librosa.display
import matplotlib.pyplot as plt
import mido
import note_seq
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.layers import (LSTM,Bidirectional, Dense, Dropout,
                          Input, Layer)
from keras.models import Sequential
from scipy.io import wavfile
import ffmpeg

hop_length_in = 512
n_bins_in = 252
bins_octaves_in = 36
win_step = 0.01
number_notes = 88
num_cep_def = 40
num_filt_def = 40
length_per_file = 4000000
mini_batch_size, num_epochs = 128, 50
input_size = 252
number_units = 256
number_layers =3
number_classes = 88
best_accuracy = 0
size_samples = 100
contador_bad = 0
np.random.seed(400) 

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

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1)
                               ,initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1)
                               ,initializer="normal")
                               
        super(attention,self).build(input_shape)


    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)


def get_model(option):
    if option == "Base Model":
        model = Sequential()

        model.add(Input(shape=(size_samples, input_size) ))
        model.add(LSTM(number_units,return_sequences = "True",kernel_initializer='normal', activation='tanh'))
        model.add(Dropout(0.2))
        for i in range(number_layers - 1):
            model.add(LSTM(number_units,return_sequences = "True",kernel_initializer='normal', activation='tanh'))
            model.add(Dropout(0.2))
        model.add(Dense(number_classes, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile()
        model.build(input_shape = (size_samples, input_size))
        model.load_weights(os.path.join("weights", "weights_LSTM.hdf5"))
        model.summary()
        return model
    
    if option =="Attention-BiLSTM":
        model = Sequential()

        ## --------------- ENCODER ----------------##
        model.add(Input(shape=(size_samples, input_size)))
        model.add(attention(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(number_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(0.2))
        ## --------------- DECODER ----------------##
        # model.add(attention(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(number_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(0.2))
        # model.add(attention(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(number_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(0.2))
        model.add(Dense(number_units, activation="tanh"))
        model.add(Dense(number_classes, activation='sigmoid'))
        model.compile() 
        model.build(input_shape=(size_samples, input_size))
        model.load_weights(os.path.join("weights", "weights_attention-Bidirectional-LSTM.hdf5"))
        model.summary()
        return model
    elif option =="MultiheadAttention-BiLSTM":
        model = Sequential()

        ## --------------- ENCODER ----------------##
        model.add(Input(shape=(size_samples, input_size)))
        model.add(attention_multi(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(number_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(0.2))
        ## --------------- DECODER ----------------##
        # model.add(attention(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(number_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(0.2))
        # model.add(attention(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(number_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(0.2))
        model.add(Dense(number_units, activation="tanh"))
        model.add(Dense(number_classes, activation='sigmoid'))
        model.compile() 
        model.build(input_shape=(size_samples, input_size))
        model.load_weights(os.path.join("weights", "weights_multihead_Bidirectional-LSTM.hdf5"))
        model.summary()
        return model
    
    elif option =="MultiheadAttention-BiLSTM(BCE)":
        model = Sequential()

        ## --------------- ENCODER ----------------##
        model.add(Input(shape=(size_samples, input_size)))
        model.add(attention_multi(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(number_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(0.2))
        ## --------------- DECODER ----------------##
        # model.add(attention(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(number_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(0.2))
        # model.add(attention(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(number_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(0.2))
        model.add(Dense(number_units, activation="tanh"))
        model.add(Dense(number_classes, activation='sigmoid'))
        model.compile() 
        model.build(input_shape=(size_samples, input_size))
        model.load_weights(os.path.join("weights", "weights_multihead_Bidirectional-LSTM(BCE).hdf5"))
        model.summary()
        return model
    
    elif option =="MultiheadAttention-BiLSTM(Single)":
        model = Sequential()

        ## --------------- ENCODER ----------------##
        model.add(Input(shape=(size_samples, input_size)))
        model.add(attention_multi(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(number_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(0.2))
        ## --------------- DECODER ----------------##
        # model.add(attention(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(number_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(0.2))
        # model.add(attention(return_sequences=True)) # receive 3D and output 3D
        model.add(Bidirectional(LSTM(number_units, return_sequences="True", activation="tanh")))
        model.add(Dropout(0.2))
        model.add(Dense(number_units, activation="tanh"))
        model.add(Dense(number_classes, activation='sigmoid'))
        model.compile() 
        model.build(input_shape=(size_samples, input_size))
        model.load_weights(os.path.join("weights", "weights_multihead_Bidirectional-LSTM(Single).hdf5"))
        model.summary()
        return model

def predict(model, cqt_feat, labels = []):
    # Extract mfcc_features
     #### LABELING ####
    # Number of frames in the file
    train2mat = []
    while (len(train2mat) + len(cqt_feat)) >= length_per_file:
        size_to_add = length_per_file - len(train2mat)
        # Append to add to npz
        train2mat.extend(cqt_feat[0:size_to_add,:])
        # Append the labels 
        train2mat = np.array(train2mat)
        # Plotting stuff
        np.save('test' , train2mat)
        contador = contador + 1
        train2mat = []
        cqt_feat = cqt_feat[size_to_add:,:]
    if len(cqt_feat) == length_per_file:
        # Append to add to npz
        train2mat.extend(cqt_feat)
        # Append the labels 
        train2mat = np.array(train2mat)
        # Plotting stuff

        np.save('{}_X'.format('test' ), train2mat)
        contador = contador + 1
        train2mat = []
    elif len(cqt_feat) > 0:
        # Append to add to npz
        train2mat.extend(cqt_feat)
        # Append the labels 

    input_array = np.array(train2mat)
    print(input_array.shape)
    print ("Predicting model. . . ")
    max_shape = (input_array.shape[0]//100)*100 
    input_array = np.array(np.reshape(input_array[0:max_shape,:],(input_array.shape[0]//size_samples,size_samples,input_size)))
    with tf.device('/cpu:0'):
        input_array = tf.convert_to_tensor(input_array, np.float32)
    predictions = model.predict(input_array, batch_size=mini_batch_size, verbose = 1)
    predictions = np.array(predictions).round()
    predictions[predictions > 1] = 1
    predictions = np.reshape(predictions,(predictions.shape[0]*predictions.shape[1],predictions.shape[2]))

    predictions = np.array(predictions).round()
    predictions[predictions > 1] = 1

    for a in range(predictions.shape[1]):
        for j in range(2,predictions.shape[0]-3):
            if predictions[j-1,a] == 1 and predictions[j,a] == 0 and predictions[j+1,a] == 0 and predictions[j+2,a] == 1:
                predictions[j,a] = 1
                predictions[j+1,a] = 1
            if predictions[j-2,a] == 0 and predictions[j-1,a] == 0 and predictions[j,a] == 1 and predictions[j+1,a] == 1 and predictions[j+2,a] == 0 and predictions[j+3,a] == 0:
                predictions[j,a] = 0
                predictions[j+1,a] = 0
            if predictions[j-1,a] == 0 and predictions[j,a] == 1 and predictions[j+1,a] == 0 and predictions[j+2,a] == 0:
                predictions[j,a] = 0
            if predictions[j-1,a] == 1 and predictions[j,a] == 0 and predictions[j+1,a] == 1 and predictions[j+2,a] == 1:
                predictions[j,a] = 1

    print(predictions.shape)
    plot_predictions(predictions, labels)
    mid_new = arry2mid(predictions)
    if not os.path.exists("ouputs"):
            os.makedirs("ouputs")
    mid_new.save('./ouputs/output.mid')
    midi_file_path = "./ouputs/output.mid"
    st.markdown(get_download_link("./ouputs/output.mid", "Download Predictions"), unsafe_allow_html=True)
    return midi_file_path, predictions

def get_download_link(file_name, link_text):
    """Generates a download link."""
    with open(file_name, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/midi;base64,{b64}" download="{file_name}">{link_text}</a>'
    return href

def arry2mid(ary):
    # get the difference
    new_ary = np.concatenate([np.array([[0] * 88]), np.array(ary)], axis=0)
    changes = new_ary[1:] - new_ary[:-1]
    # create a midi file with an empty track
    mid_new = mido.MidiFile()
    track = mido.MidiTrack()
    mid_new.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=5600000, time=0))
    # add difference in the empty track
    last_time = 0
    for ch in changes:
        if set(ch) == {0}:  # no change
            last_time += 1
        else:
            on_notes = np.where(ch > 0)[0]
            on_notes_vol = ch[on_notes]
            off_notes = np.where(ch < 0)[0]
            first_ = True
            for n, v in zip(on_notes, on_notes_vol):
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_on', note=n + 21, velocity=int(v+49), time=new_time))
                first_ = False
            for n in off_notes:
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_off', note=n + 21, velocity=0, time=new_time))
                first_ = False
            last_time = 0
    return mid_new

def get_score(predictions, labels):
    minus = labels.shape[0] - predictions.shape[0]
    labels = labels[:labels.shape[0] - minus, :]
    TP = np.count_nonzero(np.logical_and( predictions == 1, labels == 1 ))
    FN = np.count_nonzero(np.logical_and( predictions == 0, labels == 1 ))
    FP = np.count_nonzero(np.logical_and( predictions == 1, labels == 0 ))
    if (TP + FN) > 0:
        R = TP/float(TP + FN)
        P = TP/float(TP + FP)
        A = 100*TP/float(TP + FP + FN)
        if P == 0 and R == 0:
            F = 0
        else: 
            F = 100*2*P*R/(P + R)
    else: 
        A = 0
        F = 0
        R = 0
        P = 0

    return F, A, P, R

def plot_predictions(predictions, labels):
    fig, ax = plt.subplots(1, figsize=[10, 3])
    ax.imshow(predictions.transpose(), cmap='Greys', aspect='auto', interpolation='nearest')
    ax.set_title("Predictions")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Features")
    st.pyplot(fig)
    if labels != []:
        fig, ax = plt.subplots(1, figsize=[10, 3])
        ax.imshow(labels.transpose(), cmap='Greys', aspect='auto', interpolation='nearest')
        ax.set_title("Ground Truth")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Features")
        st.pyplot(fig)

st.title("Music Transcription")


# Function to handle WAV file upload
def upload_wav():
    st.subheader("Upload a WAV file")
    wav_file = st.file_uploader("Upload WAV", type=["wav","mp3","mp4"])
    if wav_file is not None:
        file_extention = os.path.splitext(wav_file.name)[1]
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        with open(os.path.join("uploads", f"audio{file_extention}"), "wb") as f:
            f.write(wav_file.getbuffer())
        st.success("WAV file uploaded successfully!")
        if file_extention != ".wav":
            if os.path.isfile(os.path.join("uploads", "audio.wav")):
                os.remove(os.path.join("uploads", "audio.wav"))
            ffmpeg.input(os.path.join("uploads", f"audio{file_extention}")).output(os.path.join("uploads", "audio.wav")).run()
    return wav_file

# Helper function to display WAV file information
def display_wav_info(file_name):
    """Displays information about the uploaded WAV file."""
    file_path = os.path.join("uploads", file_name)
    sample_rate, data = wavfile.read(file_path)
    st.write(f"Sample Rate: {sample_rate} Hz")
    try:
        st.write(f"Number of Channels: {data.shape[1]}")
        chanel_num = data.shape[1]
    except:
        st.write(f"Number of Channels: 1")
        chanel_num = 1
    st.write(f"Duration: {data.shape[0] / sample_rate} seconds")
    return sample_rate, data, chanel_num

# Function to handle MIDI file upload
def upload_midi():
    st.subheader("Upload a MIDI file for verification")
    midi_file = st.file_uploader("Upload MIDI", type=["mid", "midi"])
    if midi_file is not None:
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        with open(os.path.join("uploads", "sample.mid"), "wb") as f:
            f.write(midi_file.getbuffer())
        st.success("MIDI file uploaded successfully!")
    return midi_file

def make_label(vector_aux,number_Frames,number_notes):
    labels = None
    ns = note_seq.midi_file_to_sequence_proto(os.path.join("uploads", "sample.mid"))
    name = "sample"
    start_time = [i.start_time for i in ns.notes]
    end_time = [i.end_time for i in ns.notes]
    pitch = [i.pitch for i in ns.notes]
    text = open(f'./uploads/{name}.txt','w')
    text.write('OnsetTime\tOffsetTime\tMidiPitch\n')
    for i in range(len(pitch)):
        text.write(f'{start_time[i]:.6f}\t{end_time[i]:.6f}\t{pitch[i]}\n')
    text.close()
    labels = np.zeros((number_Frames, number_notes)).astype(np.uint8)
    file = open("./uploads/sample.txt" , "r")
    for line in file: 
        line_split = line.split()
        if line_split[0] == "OnsetTime":
            pass
        else:
            init_range, fin_range, pitch = float(line_split[0]), float(line_split[1]), int(line_split[2])
            pitch = pitch - 21
            index_min = np.where(vector_aux >= init_range)
            index_max = np.where(vector_aux - 0.01 > int((fin_range)*100)/float(100))
            try:
                labels[index_min[0][0]:index_max[0][0],pitch] = 1
            except:
                index_max = np.where(vector_aux > int((fin_range)*100)/float(100))
                try:
                    labels[index_min[0][0]:index_max[0][0],pitch] = 1
                except:
                    index_max = np.where(vector_aux + 0.01> int((fin_range)*100)/float(100))
                    labels[index_min[0][0]:index_max[0][0],pitch] = 1
    file.close()

    return labels

# Create buttons for uploading WAV and MIDI files

sampling_freq, stereo_vector = None, None
win_len = None 
mono_vector = None
cqt_feat = None 
#### LABELING ####
# Number of frames in the file
number_Frames = None
# Aux_Vector of times
vector_aux = None
options = ['Base Model','Attention-BiLSTM', 'MultiheadAttention-BiLSTM', 'MultiheadAttention-BiLSTM(BCE)','MultiheadAttention-BiLSTM(Single)']

# Create the dropdown box
selected_option = st.selectbox("Select Model:", options)
st.write("Upload your audio files:")
wav_file = upload_wav()
if (wav_file is not None):
    sampling_freq, stereo_vector ,chanel_num = display_wav_info("audio.wav") 
    win_len = 512/float(sampling_freq)
    if chanel_num == 1:
        mono_vector = stereo_vector.astype(np.float32)
    else:
        mono_vector = np.mean(stereo_vector, axis = 1)
    cqt_feat = np.absolute(librosa.cqt(mono_vector,sr = sampling_freq, hop_length = hop_length_in, n_bins = n_bins_in, bins_per_octave = bins_octaves_in)).transpose().astype(np.float32)
    number_Frames = np.max( cqt_feat.shape[0])
    vector_aux = np.arange(1, number_Frames + 1)*win_len
sample_midi = upload_midi()


if st.button("Predict"):
    # if (midi_file_path is not None):
    #     play_midi(midi_file_path)
    if (wav_file is None):
        text = '<div style="text-align:center; padding:10px; background-color:#caf0f8; border-radius:5px;"><b>Accuracy: No Audio File to predict</b></div>'
    elif (sample_midi and wav_file is not None):
        label = make_label(vector_aux,number_Frames, number_notes)
        midi_file_path, predictions = predict(get_model(selected_option), cqt_feat, labels = label)
        F, A, P, R  = get_score(predictions, label)
        text = '<div style="text-align:center; padding:10px; background-color:#caf0f8; border-radius:5px;"><b>Accuracy: P:{:.2f} - R:{:.2f} - F:{:.2f} - A:{:.2f}</b></div>'.format(P,R,F,A)
    else:
        midi_file_path, predictions = predict(get_model(selected_option), cqt_feat)
        text = '<div style="text-align:center; padding:10px; background-color:#caf0f8; border-radius:5px;"><b>Accuracy: No Midi File for calculate Score</b></div>'

        # Display the green box
    st.markdown(text, unsafe_allow_html=True)
