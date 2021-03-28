import os
import chess
import chess.pgn
import numpy as np
from tensorflow import keras
import tensorflow as tf
model = keras.models.load_model('D:\ChessModels\Chess9.2')
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
        opmove=1
        if engine_Moves.index(move)%2==1:
            opmove=-1
        clonemove=move
        if 'x' in move:
            mod = 1
            x_Pos = move.index('x')
            if '#' in move:
                move=move.replace("#","")
                baseValue += 999 * mod * opmove
            if '+' in move:
                move = move.replace("+","")
                baseValue += 1 * mod * opmove
            piecePos = boardIndex(move[x_Pos + 1::])
            piece = chess_board.piece_at(piecePos)
            if engine_Moves[length-1]==move:
                mod=0.5
            baseValue += pieceValue(piece)*mod*opmove
                # print('hell')
        chess_board.push_san(str(clonemove))
    # print (engine_Moves)
    # print (baseValue)
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

def evalBoard(position,player):
    score = 0
    if position.is_checkmate():
        if (player):
            return -9999
        else:
            return 9999
    for i in range(64):
        x=str(position.piece_at(i))
        if (x=="P"):
            score+=1
        elif (x=="p"):
            score-=1
        elif (x=="R"):
            score+=5
        elif (x=="r"):
            score-=5
        elif (x=="N"):
            score+=3
        elif (x=="n"):
            score-=3
        elif (x=="B"):
            score+=3
        elif (x=="b"):
            score-=3
        elif (x=="Q"):
            score+=9
        elif (x=="q"):
            score-=9
    return score


def minimax(position, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or position.is_game_over():
        return None, evalBoard(position,maximizingPlayer)

    legal_move = str(position.legal_moves)[38:-2].replace(',', '').split()
    bestMove=legal_move[0]
    if maximizingPlayer:
        maxEval = -99999
        for move in legal_move:
            board2 = position.copy()
            board2.push_san(move)
            eval = minimax(board2, depth - 1, alpha, beta, False)[1]
            if eval>maxEval:
                maxEval = eval
                bestMove=move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return bestMove, maxEval

    else:
        minEval = +99999
        for move in legal_move:
            board2 = position.copy()
            board2.push_san(move)
            eval = minimax(board2, depth - 1, alpha, beta, True)[1]
            if eval < minEval:
                minEval = eval
                bestMove = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return bestMove, minEval

def eval_Branches(moveTree,chess_board):
    # always prunes last layer
    maximize=1
    y=1
    highest=-9999
    lowest=9999
    #alpha beta pruning
    alpha=99999
    beta=99999
    prune=0

    highest_moveset=[]
    lowest_moveset=[]
    prevmoveset=[]
    new_movetree=[]
    #condtion
    while len(moveTree)>1:
        #loops through the list of movetrees
        for x in range(len(moveTree)):
            #finds if move is in same node as previous O(1)
            if ((moveTree[x][:len(moveTree[x])-y]==prevmoveset[:len(moveTree[x])-y] and prune==0) or prevmoveset==[]):
                #finds final value of the movetree O(n)
                value = moveValue(moveTree[x],chess_board.copy(),'none')
                prevmoveset=moveTree[x]
                if maximize==1 and value>highest:
                    highest=value
                    highest_moveset=moveTree[x]
                elif maximize==0 and value<lowest:
                    lowest=value
                    lowest_moveset=moveTree[x]
            else:
                value = moveValue(moveTree[x],chess_board.copy(),'none')
                if maximize == 1:
                    new_movetree.append(highest_moveset)
                    highest_moveset=moveTree[x]
                    highest = value
                elif maximize == 0:
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
        # print(maximize)
        # print("highest",highest_moveset)
        # print("lowest",lowest_moveset)
        # print (moveTree)
        highest = -9999
        lowest = 9999
        highest_moveset = []
        lowest_moveset = []
    return moveTree



chess_board = chess.Board()
maximize=True
currentVal=-99999
turncount=0
while (not chess_board.is_stalemate() and not chess_board.is_insufficient_material() and not chess_board.is_game_over() and not chess_board.is_seventyfive_moves()):
    maximize = True
    currentVal = -99999
    print(chess_board)
    matrix = make_matrix(chess_board)
    print(evalBoard(chess_board,maximize))
    board = translate(matrix,chess_dict)
    board = np.array(board)
    board = np.reshape(board,(1,8,8,12))
    legal_moves = str(chess_board.legal_moves)[38:-2].replace(',','').split()
    print (legal_moves)
    pred = model.predict(board)
    index = int(round((len(legal_moves)*pred)[0][0]))
    try:
        engine_Move=legal_moves[index]
    except:
        engine_Move = legal_moves[len(legal_moves)-1]
    boardEngine = chess_board.copy()
    boardEngine.push_san(engine_Move)
    engine_value=minimax(boardEngine, 3, -99999, +99999, not maximize)[1]
    boardMini = chess_board.copy()
    minmaxMove,val2 = minimax(boardMini, 4, -99999, +99999, maximize)

    print(maximize)
    print(minmaxMove)
    print(val2)
    print(engine_Move)
    print(engine_value)

    if ((maximize and val2>engine_value or not maximize and val2<engine_value) or turncount>5):
        move=minmaxMove
        print (move)
        chess_board.push_san(move)
    else:
        chess_board.push_san(engine_Move)
        print(engine_Move)
    #selfplay
    # if (maximize):
    #     maximize=False
    #     currentVal=99999
    # elif (not maximize):
    #     maximize=True
    #     currentVal=-99999
    if chess_board.is_checkmate():
        break
    print(chess_board)
    while True:
        print(str(chess_board.legal_moves)[38:-2].replace(',','').split())
        val = input("Enter your move: ")
        try:
            chess_board.push_san(val)
            break
        except:
            pass
    turncount+=1
    # except:
    #      print("try again invalid")
    #      pass