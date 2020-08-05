#!/usr/bin/python3
import sys
import os
import glob
import librosa
from numpy import save
import numpy as np
import soundfile as sf
from numpy import load

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers import LSTM, Dense

def Call_Back():
    checkpoint_path = 'Brains/brain_' + str(len(glob.glob('Brains/*.ckpt'))) + '.ckpt'
    return ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose = 1)

def Load(x):
    return load(glob.glob('Ultimate/' + x + '*.npy')[0])

def Train():
    print('Training:')
    x_train = Load("x_train")
    y_train = Load("y_train")
    x_val = Load("x_val")
    y_val = Load("y_val")
    #x_test = Load("x_test")
    #y_test = Load("y_test")

    model = Sequential()
    model.add(LSTM(2050, input_shape=(1,x_train.shape[-1]), return_sequences=True))
    model.add(LSTM(2050, return_sequences = True))
    model.add(Dense(2050))
    model.compile(loss='mse', optimizer='rmsprop', metrics= ['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), callbacks = [Call_Back()])

    model.save('Brains/brain_' + str(len(glob.glob('Brains/*.h5'))) + '.h5')
    print(model.got_weights())

def Load_model(x):
    return load_model(glob.glob('Brains/brain_' + x + '.h5')[0])

def Ultimate(x):
    print('Ultimate:')
    files = glob.glob(x + "/*.npy")
    Data = np.array(load(files[0]))
    for file in files[1:]:
        Data = np.append(Data, load(file), axis = 0)
        print('\t' + file.split('/')[1])
    print('\t' + str(Data.shape))
    Data = Data.reshape(Data.shape[0], 1, Data.shape[-1])
    n = len(Data)

    x_train = Data[:n // 2]
    save('Ultimate/x_train' + '_' + str(len(x_train)) + '.npy', x_train)
    print('\t' + str(x_train.shape))

    y_train = Data[1:n // 2 + 1]
    save('Ultimate/y_train' + '_' + str(len(y_train)) + '.npy', y_train)
    print('\t' + str(y_train.shape))

    x_val = Data[n // 2 : n // 2 + n // 4]
    save('Ultimate/x_val' + '_' + str(len(x_val)) + '.npy', x_val)
    print('\t' + str(x_val.shape))

    y_val = Data[n // 2 + 1 : n // 2 + n // 4 + 1]
    save('Ultimate/y_val' + '_' + str(len(y_val)) + '.npy', y_val)
    print('\t' + str(y_val.shape))
    
    x_test = Data[n // 2 + n // 4 : -1]
    save('Ultimate/x_test' + '_' + str(len(x_test)) + '.npy', x_test)
    print('\t' + str(x_test.shape))

    y_test = Data[n // 2 + n // 4 + 1 :]
    save('Ultimate/y_test' + '_' + str(len(y_test)) + '.npy', y_test)
    print('\t' + str(y_test.shape))

def Restore(x):
    print('Restore:')
    #load
    files = glob.glob(x + "/*.npy")
    for file in files:
        Data = load(file)

        #back to complex
        Btc = []
        for X in range(len(Data)):
            line = []
            for Y in range(int(len(Data[X])/2)):
                line.append(complex(Data[X][Y*2],Data[X][Y*2+1]))
            Btc.append(line)
        Btc = np.asarray(Btc)
        Btc = Btc.T

        #invert
        Btc = librosa.istft(Btc, length = len(Data) * 512)

        # Write out audio as 24bit PCM WAV
        sf.write('Output/data_' + str(files.index(file)) + '.wav', Btc, 22050, subtype='PCM_24')
        print('\t' + file.split('/')[1])

def Pre_processing():
    print('Pre_processing:')
    files = glob.glob('Source/*.wav')
    for file in files:
        print('\t' + file.split('/')[1])
        # load audio file with Librosa
        signal, sample_rate = librosa.load(file, sr=22050)

        # STFT -> spectrogram
        hop_length = 512 # in num. of samples
        n_fft = 2048 # window in num. of samples

        # perform stft
        stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

        # transform
        stft = stft.T

        # Separate
        STstft = []
        for X in range(len(stft)):
            line = []
            for Y in range(len(stft[X])):
                line.append(stft[X][Y].real)
                line.append(stft[X][Y].imag)
            STstft.append(line)
        STstft = np.asarray(STstft)
        stft = []

        #save
        save('Pre-processed/data_' + str(files.index(file)) + '_' + str(len(STstft)) + '.npy', STstft)

def Debug():
    pass

def Youtube():
    os.system('cd Source/; youtube-dl -a urls.txt -x --audio-format wav --audio-quality 0')

def Remove(x, y):
    print('Remove all:')
    files = glob.glob(x + '/*.' + y)
    for file in files:
        os.remove(file)
        print('\t' + file)

def main(argv):
    for opt in argv:
        if opt == '-h':
            print('test.py [-pa, -pd, -pr, -y, -d]')
        elif opt == '-p':
            Pre_processing()
        elif opt == '-pd':
            Remove('Pre-processed', 'npy')
        elif opt == '-pr':
            Restore("Pre-processed")
        elif opt == '-y':
            Youtube()
        elif opt == '-yd':
            Remove('Source', 'wav')
        elif opt == '-d':
            Debug()
        elif opt == '-u':
            Ultimate("Pre-processed")
        elif opt == '-t':
            Train()
    sys.exit()

if __name__ == "__main__":
   main(sys.argv[1:])