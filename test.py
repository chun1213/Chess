import os
import chess
import chess.pgn
import numpy as np
from tensorflow import keras
import tensorflow as tf
model = keras.models.load_model('D:\ChessModels\Chess9.1')
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
            terms.append(chess_dict[term])
        rows.append(terms)
    return rows

def boardIndex(position):
    index=0;
    letter=position[0:1:]
    number = int(position[1::])
    if letter=='b':
        index+=1
    elif letter=='c':
        index += 2
    elif letter=='d':
        index += 3
    elif letter=='e':
        index += 4
    elif letter=='f':
        index += 5
    elif letter=='g':
        index += 6
    elif letter=='h':
        index += 7
    return index+((number-1)*8)

def pieceValue(letter):
    value=0
    letter=str(letter)
    if letter=='p' or letter=='P':
        value=1
    elif letter=='b' or letter=='B':
        value=3
    elif letter=='n' or letter=='N':
        value=3
    elif letter=='r' or letter=='R':
        value=5
    elif letter=='q' or letter=='Q':
        value=9
    return value

def moveValue(engine_Moves,chess_board,initial_move):
    baseValue = 0
    length=len(engine_Moves)
    if initial_move=='none':
        pass
    else:
        chess_board.push_san(initial_move)
    opmove=0
    for move in engine_Moves:
        opmove=0
        if engine_Moves.index(move)%2==1:
            opmove=1
        clonemove=move
        if 'x' in move:
            x_Pos = move.index('x')
            move=move.replace("#","")
            move = move.replace("+","")
            piecePos = boardIndex(move[x_Pos + 1::])
            piece = chess_board.piece_at(piecePos)
            mod=1
            if engine_Moves[length-1]==move:
                mod=0.5
            if opmove==1:
                baseValue -= pieceValue(piece)*mod
                print('hell')
            else:
                baseValue += pieceValue(piece)*mod
                print('hell0')
        chess_board.push_san(str(clonemove))
    print (engine_Moves)
    print (baseValue)
    return baseValue

def branching_moves(move,chess_board,depth):
    if move != 'none':
        chess_board.push_san(move)
    legal_moves = str(chess_board.legal_moves)[38:-2].replace(',', '').split()
    move_tree=[]
    for moves in legal_moves:
        move_line = []
        board2=chess_board.copy()
        if depth>0:
            for x in branching_moves(moves,board2,depth-1):
                move_line.append(moves)
                move_line+=x
                move_tree.append(move_line)
                move_line = []
        else:
            move_line.append(moves)
            move_tree.append(move_line)
    return move_tree

def eval_Branches(moveTree,chess_board):
    # always prunes last layer
    maximize=1
    y=1
    highest=-9999
    lowest=9999
    highest_moveset=[]
    lowest_moveset=[]
    prevmoveset=[]
    new_movetree=[]
    while len(moveTree)>1:
        for x in range(len(moveTree)):
            if moveTree[x][:len(moveTree[x])-y]==prevmoveset[:len(moveTree[x])-y] or prevmoveset==[]:
                value = moveValue(moveTree[x],chess_board.copy(),'none')
                prevmoveset=moveTree[x]
                if maximize==1 and value>highest:
                    highest=value
                    highest_moveset=moveTree[x]
                if maximize==0 and value<lowest:
                    lowest=value
                    lowest_moveset=moveTree[x]
            else:
                value = moveValue(moveTree[x],chess_board.copy(),'none')
                if maximize == 1:
                    new_movetree.append(highest_moveset)
                    highest_moveset=moveTree[x]
                    highest = value
                if maximize == 0:
                    new_movetree.append(lowest_moveset)
                    lowest_moveset = moveTree[x]
                    lowest = value
                prevmoveset=moveTree[x]
        y+=1
        if maximize == 1:
            new_movetree.append(highest_moveset)
            maximize = 0
        elif maximize == 0:
            new_movetree.append(lowest_moveset)
            maximize = 1
        moveTree=new_movetree
        new_movetree=[]
        prevmoveset = []
        print(maximize)
        print("highest",highest_moveset)
        print("lowest",lowest_moveset)
        print (moveTree)
        highest = -9999
        lowest = 9999
        highest_moveset = []
        lowest_moveset = []
    return moveTree



chess_board = chess.Board()
while (not chess_board.is_stalemate() and not chess_board.is_insufficient_material() and not chess_board.is_game_over() and not chess_board.is_seventyfive_moves()):
    print(chess_board)
    # board2=chess_board.copy()
    # board2.push_san('e4')
    # board2.push_san('d5')
    # print (eval_Branches(branching_moves('none',board2.copy(),2),board2))
    # print (chess_board.piece_at(boardIndex('e1')))
    val = input("Enter your move: ")
    try:
        chess_board.push_san(val);
        print(chess_board.legal_moves)
        matrix = make_matrix(chess_board)
        board = translate(matrix,chess_dict)
        board = np.array(board)
        board = np.reshape(board,(1,8,8,12))
        legal_moves = str(chess_board.legal_moves)[38:-2].replace(',','').split()
        pred = model.predict(board)
        print (pred)
        index = int(round((len(legal_moves)*pred)[0][0]))
        engine_Move=legal_moves[index]
        print(engine_Move)
        board2=chess_board.copy()
        board2.push_san(engine_Move)
        val2=moveValue((eval_Branches(branching_moves('none',board2.copy(),2),board2))[0],board2,'none')
        print (val2)
        if (val2>1):
            move=(eval_Branches(branching_moves('none',chess_board.copy(),2),chess_board.copy()))[0]
            print (move)
            chess_board.push_san(move[0])
        else:
            chess_board.push_san(engine_Move)
        if chess_board.is_checkmate():
            break
    except:
         print("try again invalid")
         pass