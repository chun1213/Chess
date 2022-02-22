import os
import chess
import chess.pgn
import numpy as np
import pickle
import pandas as pd
from sklearn.utils import class_weight
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
os.chdir('D:\Downloads\lichess_elite_2021-12')

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

counter = 1
for filename in os.listdir(os.getcwd()):
    print ("file is",filename)
    pgn = open(filename)
    first_game = chess.pgn.read_game(pgn)
    while first_game is not None:
        moves=str(first_game.mainline_moves()).split()
        del moves[::3]
        str_moves=" ".join(str(x) for x in moves)
        first_game = chess.pgn.read_game(pgn)
        print (counter)
        counter+=1
        data.append(str_moves)
        if len(data) > 30000: # max games in data
            break2=1
            break
    if break2 ==1:
        break



# os.chdir('D:\Downloads\chess')
# df = pd.read_csv('games.csv')
# data = df['moves'].tolist()
# print (data)



lendata= len(data)
data=data[:len(data)-1]


# data = data[:1000]

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

def translate(matrix,chess_dict,color):
    rows = []
    for row in matrix:
        terms = []
        for term in row:
            terms.append(chess_dict[term])
        turn = [0] * 12
        if color == chess.BLACK:
            turn[1] = 1
        else:
            turn[0] = 1
        terms.append(turn)
        rows.append(terms)
    tuples = tuple(tuple(j) for i in rows for j in i )

    return rows,tuples

def create_Output(insquare,outsquare,promo,kingcastle,queencastle):
    out = [0] * 134
    if kingcastle:
        out[132] = 1
        return out
    elif queencastle:
        out[133] = 1
        return out
    #insquare
    out[insquare] = 1
    #outsquare
    out[64+outsquare] = 1 #64 accounts for the offset
    # for promotions only
    if promo !=None:
        if promo == 2: #knight
            out[128] = 1
        elif promo == 3: #bishop
            out[129] = 1
        elif promo == 4: #rook
            out[130] = 1
        elif promo == 5: #queen
            out[131] = 1
    return out




def data_setup():
    X = []
    out = []
    setx=set()
    for game in range(len(data)):
        data[game] = data[game].split()
        board = chess.Board()
        print('done game')
        for move in range(len(data[game])):
            matrix = make_matrix(board.copy())
            translated,translatedset = translate(matrix,chess_dict,board.turn)
            # print(translatedset)
            # print(np.array(translated).shape)
            # if translatedset not in setx:
            print ("found")
            # setx.add(translatedset)
            X.append(translated)
            print(game)
            value=data[game][move]
            objmove = None
            try:
                objmove = board.parse_san(value)
            except:
                value = value[0: 1:] + value[2::]
                try:
                    objmove = board.parse_san(value)
                except:
                    value = value[0: 1:] + value[2::]
                    objmove = board.parse_san(value)

            start_square = objmove.from_square
            end_square = objmove.to_square
            print(start_square)
            print(end_square)
            kingcastle = False
            queencastle = False
            if "O-O" in value:
                kingcastle = True
            elif "O-O-O" in value:
                queencastle = True
            out.append(create_Output(start_square,end_square,objmove.promotion,kingcastle,queencastle))
            board.push_san(data[game][move])




    return X,out

earlystop_callback = EarlyStopping(
  monitor='loss', min_delta=0.00001,
  patience=100, restore_best_weights=True)


def initialize_network():
    model = Sequential()
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', input_shape=(8,9,12),padding='same'))
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
    model.add(Dense(134,activation = 'sigmoid')) #change
    return model

def my_loss(weight):
    def weighted_cross_entropy_with_logits(labels, logits):
        loss = tf.nn.weighted_cross_entropy_with_logits(
            tf.cast(labels, tf.float32), logits, weight
        )
        return loss
    return weighted_cross_entropy_with_logits


def findKthLargest( nums, k):
    nums.sort()
    if k == 1:
        return nums[-1]
    temp = 1
    return nums[len(nums) - k]

def newmove(largest1,largest2,fhalf,shalf):
    largest2 += 1
    bestmove1 = findKthLargest(fhalf.copy(), largest1)
    bestmove2 = findKthLargest(shalf.copy(), largest2)
    if bestmove2 < 0.8:
        largest2 = 1
        largest1 +=1
        bestmove2 = findKthLargest(shalf.copy(), largest2)
        bestmove1 = findKthLargest(fhalf.copy(), largest1)
    bestmove2i = np.where(shalf == bestmove2)[0][0]
    bestmove1i = np.where(fhalf == bestmove1)[0][0]
    return bestmove1i,bestmove2i,largest1,largest2


def translateback(tensor,board):
    fhalf = tensor[:64]
    shalf = tensor[64:128]
    promo = tensor[128:132]
    castle1 = tensor[132]
    castle2 = tensor[133]
    bestmove1 = (max(fhalf))
    bestmove2 = (max(shalf))
    bestmove1i = np.where(fhalf == bestmove1)[0][0]
    bestmove2i = np.where(shalf == bestmove2)[0][0]
    largest1=1
    largest2=1
    while (bestmove1 + bestmove2)/2 > castle1 and (bestmove1 + bestmove2)/2 > castle2:
        if max(promo) > 0.8:
            index=np.where(promo == max(promo))[0][0]
            if index == 0:
                try:
                    move = board.parse_uci(chess.square_name(bestmove1i)+chess.square_name(bestmove2i)+"k")
                    print(chess.square_name(bestmove1i), "to", chess.square_name(bestmove2i),"k")
                    return move
                except:
                    bestmove1i,bestmove2i,largest1,largest2=newmove(largest1,largest2,fhalf,shalf)

            elif index == 1:
                try:
                    move = board.parse_uci(chess.square_name(bestmove1i) + chess.square_name(bestmove2i) + "b")
                    print(chess.square_name(bestmove1i), "to", chess.square_name(bestmove2i), "b")
                    return move
                except:
                    bestmove1i,bestmove2i,largest1,largest2=newmove(largest1,largest2,fhalf,shalf)
            elif index == 2:
                try:
                    move = board.parse_uci(chess.square_name(bestmove1i) + chess.square_name(bestmove2i) + "r")
                    print(chess.square_name(bestmove1i), "to", chess.square_name(bestmove2i), "r")
                    return move
                except:
                    bestmove1i,bestmove2i,largest1,largest2=newmove(largest1,largest2,fhalf,shalf)
            elif index == 3:
                try:
                    move = board.parse_uci(chess.square_name(bestmove1i) + chess.square_name(bestmove2i) + "q")
                    print(chess.square_name(bestmove1i), "to", chess.square_name(bestmove2i), "q")
                    return move
                except:
                    bestmove1i,bestmove2i,largest1,largest2=newmove(largest1,largest2,fhalf,shalf)
        else:
            try:
                move = board.parse_uci(chess.square_name(bestmove1i) + chess.square_name(bestmove2i))
                print(chess.square_name(bestmove1i), "to", chess.square_name(bestmove2i))
                return move
            except:
                print("fail")
                print(shalf)
                print(fhalf)
                bestmove1i,bestmove2i,largest1,largest2=newmove(largest1,largest2,fhalf,shalf)

    if castle1 >= castle2:
        print("O-O")
        return board.parse_san("O-O")
    if castle2 > castle1:
        print("O-O-O")
        return board.parse_san("O-O-O")

# model = keras.models.load_model('D:\ChessModels\Chessmodel4.0', compile=False)
# chess_board = chess.Board()
# while (not chess_board.is_stalemate() and not chess_board.is_insufficient_material() and not chess_board.is_game_over() and not chess_board.is_seventyfive_moves()):
#     matrix = make_matrix(chess_board)
#     board,boardset = translate(matrix, chess_dict,chess_board.turn)
#     board = np.array(board)
#     board = np.reshape(board, (1, 8, 9, 12))
#     pred = model.predict(board)[0]
#     compmove = translateback(pred,chess_board)
#     print(compmove)
#     chess_board.push(compmove)
#     print(chess_board)
#
#     while True:
#         print(str(chess_board.legal_moves)[38:-2].replace(',', '').split())
#         val = input("Enter your move: ")
#         try:
#             chess_board.push_san(val)
#             break
#         except:
#             pass




print('Preparing Data...')
start = time.time()
X,y = data_setup()
# with open('inputs.pkl', 'wb') as f:
#     pickle.dump(X, f)
# with open('outputs.pkl', 'wb') as f:
#     pickle.dump(y, f)
print('done prep')
one_game_time = time.time()-start
X = np.array(X)
y = np.array(y)
X_test = X[:int(len(X)*0.1)]
X_train = X
print(len(X_train))
y_test = y[:int(len(X)*0.1)]
y_train = y
print('Initalizing Network...')
model = initialize_network()
print('Compiling Network...')

model.compile(optimizer='adam', loss=my_loss(weight=32))
# dirx = 'D:\Downloads\chess'
# os.chdir(dirx)
# h5 = 'chess' + '_best_model' + '.h5'
# checkpoint = callbacks.ModelCheckpoint(h5,
#                                            monitor='val_loss',
#                                            verbose=0,
#                                            save_best_only=True,
#                                            save_weights_only=True,
#                                            mode='auto',
#                                            period=1)
# es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5000/10)
# callback = [checkpoint,es]
# json = 'chess' + '_best_model' + '.json'
# model_json = model.to_json()
# with open(json, "w") as json_file:
#     json_file.write(model_json)

print('Training Network...')
history = model.fit(X_train,y_train,epochs = 20,verbose = 2,validation_data = (X_test,y_test),callbacks=[earlystop_callback])

model.save('D:\ChessModels\Chessmodel6.0')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

model = keras.models.load_model('D:\ChessModels\Chessmodel6.0', compile=False)
chess_board = chess.Board()
while (not chess_board.is_stalemate() and not chess_board.is_insufficient_material() and not chess_board.is_game_over() and not chess_board.is_seventyfive_moves()):
    matrix = make_matrix(chess_board)
    board,boardset = translate(matrix, chess_dict,chess_board.turn)
    board = np.array(board)
    board = np.reshape(board, (1, 8, 9, 12))
    pred = model.predict(board)[0]
    compmove = translateback(pred,chess_board)
    print(compmove)
    chess_board.push(compmove)
    print(chess_board)

    while True:
        print(str(chess_board.legal_moves)[38:-2].replace(',', '').split())
        val = input("Enter your move: ")
        try:
            chess_board.push_san(val)
            break
        except:
            pass

