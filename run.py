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

def findKthLargest( nums, k):
    nums.sort()
    if k == 1:
        return nums[-1]
    temp = 1
    index = len(nums) - k
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
    if bestmove2 ==1:
        temp=list(shalf)
        bestmove2i = [i for i, n in enumerate(temp) if n == 1][largest2-1]
    else:
        bestmove2i = list(shalf).index(bestmove2)
    if bestmove1 == 1:
        temp = list(fhalf)
        bestmove1i = [i for i, n in enumerate(temp) if n == 1][largest1 - 1]
    else:
        bestmove1i = list(fhalf).index(bestmove1)

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
                print(bestmove1i)
                print(bestmove2i)
                print(largest1)
                print(largest2)
                print(chess.square_name(bestmove1i), "to", chess.square_name(bestmove2i))
                bestmove1i,bestmove2i,largest1,largest2=newmove(largest1,largest2,fhalf,shalf)


    if castle1 >= castle2:
        print("O-O")
        return board.parse_san("O-O")
    if castle2 > castle1:
        print("O-O-O")
        return board.parse_san("O-O-O")

model = keras.models.load_model('D:\ChessModels\Chessmodel6.0', compile=False)
chess_board = chess.Board()
while (not chess_board.is_stalemate() and not chess_board.is_insufficient_material() and not chess_board.is_game_over() and not chess_board.is_seventyfive_moves()):
    print(chess_board)
    while True:
        print(str(chess_board.legal_moves)[38:-2].replace(',', '').split())
        val = input("Enter your move: ")
        try:
            chess_board.push_san(val)
            break
        except:
            pass
    matrix = make_matrix(chess_board)
    board,boardset = translate(matrix, chess_dict,chess_board.turn)
    board = np.array(board)
    board = np.reshape(board, (1, 8, 9, 12))
    pred = model.predict(board)[0]
    print(pred[:64])
    print(pred[64:128])
    compmove = translateback(pred,chess_board)
    print(compmove)
    chess_board.push(compmove)
