import os

import librosa
import numpy as np
from scipy.io import wavfile


def creat_list():
    data = ['train','val','MAPS_test','MAESTRO_test']
    Output_dir = './'

    for name in data:
        f = open(Output_dir + f'{name}.lst','w')

        for filename in os.listdir(f'./data_multiple/{name}/txt'):
            f.write(filename + '\n')

        f.close() 

def raw_to_mat():
    hop_length_in = 512
    n_bins_in = 252
    bins_octaves_in = 36
    number_notes = 88
    length_per_file = 4000000

    data = ['train.lst','val.lst','MAPS_test.lst','MAESTRO_test.lst']

    for split in data:
        source_List = split
        split = split.split('.')[0]
        source_WAV = f'./data_multiple/{split}/wav_mono/'
        source_Txt = f'./data_multiple/{split}/txt/'
        out_mat = './mat_multiple/'

        train2mat = []
        labels2mat = []
        contador = 0

        source_list_split = source_List.split('.')
        source_list_split = source_list_split[0].split('/')
        list_name = source_list_split[-1]

        file_List = open( source_List , "r")

        for filename in file_List:
            filename_split = filename.split('.txt')
            sampling_freq, stereo_vector = wavfile.read(source_WAV + filename_split[0] + '.wav')
            win_len = 512/float(sampling_freq)
            # mono_vector = np.mean(stereo_vector, axis = 1)
            cqt_feat = np.absolute(librosa.cqt(stereo_vector.astype(np.float32),sr = sampling_freq, hop_length = hop_length_in, n_bins = n_bins_in, bins_per_octave = bins_octaves_in)).transpose().astype(np.float32)
            number_Frames = np.max( cqt_feat.shape[0])
            vector_aux = np.arange(1, number_Frames + 1)*win_len
            labels = np.zeros((number_Frames, number_notes)).astype(np.uint8)

            file = open( source_Txt + filename_split[0] + '.txt' , "r")
            for line in file: 
                line_split = line.split()
                if line_split[0] == "OnsetTime":
                    print (f"Preprocessing operations for {filename_split[0]} . . .")
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
            while (len(train2mat) + len(cqt_feat)) >= length_per_file:
                size_to_add = length_per_file - len(train2mat)
                train2mat.extend(cqt_feat[0:size_to_add,:])
                labels2mat.extend(labels[0:size_to_add,:])
                train2mat = np.array(train2mat)
                labels2mat = np.array(labels2mat)
                print (" Shape of MFCC is " + str(train2mat.shape) + " - Saved in " + out_mat + list_name + '/' + str(contador) + list_name)
                print (" Shape of Labels is " + str(labels2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(contador) + list_name)
                np.save('{}_X'.format(out_mat + list_name + '/' + str(contador) + list_name ), train2mat)
                np.save('{}_y'.format(out_mat + list_name + '/' + str(contador) + list_name), labels2mat)
                contador = contador + 1
                train2mat = []
                labels2mat = []
                cqt_feat = cqt_feat[size_to_add:,:]
                labels = labels[size_to_add:,:]
            if len(cqt_feat) == length_per_file:
                train2mat.extend(cqt_feat)
                labels2mat.extend(labels)
                train2mat = np.array(train2mat)
                labels2mat = np.array(labels2mat)
                print (" Shape of MFCC is " + str(train2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(contador) + list_name)
                print (" Shape of Labels is " + str(labels2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(contador) + list_name)
                np.save('{}_X'.format(out_mat + list_name + '/' + str(contador) + list_name ), train2mat)
                np.save('{}_y'.format(out_mat + list_name + '/' + str(contador) + list_name), labels2mat)
                contador = contador + 1
                train2mat = []
                labels2mat = []
            elif len(cqt_feat) > 0:
                train2mat.extend(cqt_feat)
                labels2mat.extend(labels)

        train2mat = np.array(train2mat)
        labels2mat = np.array(labels2mat)

        print (" Shape of MFCC is " + str(train2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(contador) + list_name)
        print (" Shape of Labels is " + str(labels2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(contador) + list_name)

        np.save('{}_X'.format(out_mat + list_name + '/' + str(contador) + list_name ), train2mat)
        np.save('{}_y'.format(out_mat + list_name + '/' + str(contador) + list_name), labels2mat)


if  __name__ == '__main__':
    creat_list()
    raw_to_mat()
