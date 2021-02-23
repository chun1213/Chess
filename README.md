# Chess
 Everyone knows that chess is a complex game, some even say that there are 10^120 possible board positions. So hard coding an AI to play chess is virtually impossible. I have been playing Chess for a very long time, (since I was in grade school) and have always been interested in its evolution. As a result, I was naturally drawn towards both traditional chess engines such as stockfish and AlphaZero’s Neural network AI. 

Data:

The dataset I used to train the Neural Network part of this engine was obtained from Lichess.com where a user had put together many games played between users of 2200 Elo and above (2200 Elo is very skilled). I picked this as these users would likely play the best counter moves and openings in their games. I then preprocessed the data from the pgn files and normalized the data so each board position only had one unique move that followed it. The dataset was very large but I was only able to use 10000 different games due to TensorFlow training times increasing significantly per epoch with more games. (This is with GPU CUDA acceleration, I have a GTX 1060 6GB).

Architecture:

The engine consists of two parts, a neural net and a minimax algorithm. The neural network was created using TensorFlow with multiple Conv2d layers with 256 filters and a 3x3 kernel size inspired by the architecture of AlphaZero. The Minimax Algorithm is exactly what it sounds like, it generates and looks through all the possible moves on a move tree and then finds the best line of moves to play from the current position. The downside is that its slow so that is why the neural net is used.

How it works:

The engine works by reading the state of the board and then inputs it into the neural network to obtain the next move.  The neural net basically memorizes popular positions and countermoves so that it can effectively play and counter any similar openings it faces, this solves the issues of most traditional engines which are bad at openings without hardcode. The minimax algorithm is there to handle very nontraditional and irregular board positions where the network predicts bad moves(move that lose material without compensation). In those cases, the minimax move is played instead.

Limitations:

With the project explained, I need to get to the limitations of the engine. The main issue is that the neural net relies heavily on the training data I preprocessed being unique, so I ignored all other datapoints and countermoves to a board position once that position was already in the training data. A way to fix this would be to only add the move that ways played the most times as the correct answer rather than just adding the first. This would ensure that no “bad moves” would enter the engine’s training data. Another thing is the minimax algorithm I created; this algorithm only searches at depth 3(the next 3 possible moves after the current move) due to time constraints. This would normally be solved using Alpha-Beta pruning, but I have yet to implement this at the time of writing this.
