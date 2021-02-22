import os
import chess
import chess.pgn
import numpy as np
import pandas as pd
import keras
import time
import tensorflow as tf
from keras import callbacks, optimizers
from keras.layers import (LSTM, BatchNormalization, Dense, Dropout, Flatten,
                          TimeDistributed)
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model, model_from_json
from IPython.display import clear_output
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

flatten = lambda l: [item for sublist in l for item in sublist]
break2=0
data=[]
#replace this with the place where the lichess database is stored
os.chdir('D:\Downloads\Lichess Elite Database\Lichess Elite Database')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

for filename in os.listdir(os.getcwd()):

    pgn = open(filename)
    first_game = chess.pgn.read_game(pgn)
    while first_game is not None:
        moves=str(first_game.mainline_moves()).split()
        del moves[::3]
        str_moves=" ".join(str(x) for x in moves)
        print (str_moves)
        data.append(str_moves)
        first_game=chess.pgn.read_game(pgn)
        if len(data) > 10000:
            break2=1
            break
    if break2 ==1:
        break
lendata= len(data)
data=data[:len(data)-1]

chess_dict = {
    'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
    'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
    'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
    'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
    'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
    'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
    'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
    'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
    'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
    'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
    'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
    'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
    '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
}

def make_matrix(board):
    pgn = board.epd()
    foo = []
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo2.append('.')
            else:
                foo2.append(thing)
        foo.append(foo2)
    return foo

def translate(matrix,chess_dict):
    rows = []
    for row in matrix:
        terms = []
        for term in row:
            terms.append(tuple(chess_dict[term]))
        rows.append(tuple(terms))
    return tuple(rows)

def translate2(matrix,chess_dict):
    rows = []
    for row in matrix:
        terms = []
        for term in row:
            terms.append(chess_dict[term])
        rows.append(terms)
    return rows

def set_to_list(X):
    rows = []
    for val in X:
        terms = []
        for term in val:
            terms.append(list(term))
        rows.append((terms))
    return (rows)

def data_setup():
    X = set()
    X_train=[]
    y = []
    for game in range(len(data)):
        data[game] = data[game].split()
        board = chess.Board()
        for move in range(len(data[game])):
            legal_moves = str(board.legal_moves)[38:-2].replace(',','').split()
            matrix = make_matrix(board.copy())
            translated = translate(matrix,chess_dict)
            translated2 = translate2(matrix, chess_dict)
            board.push_san(data[game][move])
            if translated not in X:
                X_train.append(translated2)
                X.add(translated)

                print(game)
                value=data[game][move]
                try:
                    y.append(legal_moves.index(value) / len(legal_moves))
                except:
                    value = value[0: 1:] + value[2::]
                    try:
                        y.append(legal_moves.index(value) / len(legal_moves))
                    except:
                        value = value[0: 1:] + value[2::]
                        y.append(legal_moves.index(value) / len(legal_moves))




    return X_train,y

earlystop_callback = EarlyStopping(
  monitor='loss', min_delta=0.00001,
  patience=100, restore_best_weights=True)

def initialize_network():
    model = Sequential()
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', input_shape=(8,8,12),padding='same'))
    model.add(MaxPooling2D(pool_size=2, strides=None,padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=2, strides=None,padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=2, strides=None,padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu',padding='same'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1,activation = 'sigmoid'))
    return model

print('Preparing Data')
start = time.time()
X,y = data_setup()
one_game_time = time.time()-start
X = np.array(X)
print(X)
y = np.array(y)
X_test = X[:int(len(X)*0.1)]
X_train = X
y_test = y[:int(len(X)*0.1)]
y_train = y
print(len(X_train))
print('Initalizing Network')
#either load a network or use initialize_network()
#model =initialize_network()
model = keras.models.load_model('D:\ChessModels\Chess10')
print('Compiling Network')

model.compile(optimizer='adam', loss='mse')
print('Training Network')
history = model.fit(X_train,y_train,epochs = 10,verbose = 1,validation_data = (X_test,y_test),callbacks=[earlystop_callback])

#replace this with the directory u wish to save in
model.save('D:\ChessModels\Chess10')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

chess_board = chess.Board()
while (not chess_board.is_stalemate() and not chess_board.is_insufficient_material() and not chess_board.is_game_over() and not chess_board.is_seventyfive_moves()):
    matrix = make_matrix(chess_board)
    board = translate(matrix,chess_dict)
    board = np.array(board)
    board = np.reshape(board,(1,8,8,12))
    pred = model.predict(board)
    legal_moves = str(chess_board.legal_moves)[38:-2].replace(',','').split()
    index = int(round((len(legal_moves)*pred)[0][0]))
    chess_board.push_san(legal_moves[index])
    print(chess_board)
    if chess_board.is_checkmate():
        break


