<h1 align="center"> Chess Engine </h1>

<p align="center">
  <img src="https://github.com/chun1213/Chess/blob/main/images/chess.jpg" width="600" />
</p>


 Everyone knows that chess is a complex game, some even say that there are 10^120 possible board positions. So hard coding an AI to play chess is virtually impossible. I have been playing Chess for a very long time, (since I was in grade school) and have always been interested in its evolution. As a result, I was naturally drawn towards both traditional chess engines such as stockfish and AlphaZero’s Neural network AI. 


<h1> Engine Use: </h1>
You will have to download the repo to use the engine, run this engine by opening the "test.py" file, the relative path should already be calculated by the file itelf but if not, simply change the path shown here on line 13 to the path of the Chess9.2 folder inside the Models folder.

<b> You will need tensorflow to be installed on your system to use the engine </b>
```python 
model = keras.models.load_model(my_absolute_dirpath+'\Models\Chess9.2')
```

<h1> Some Openings </h1>
You can play the engine as black or white, here are some opening moves the engine makes:
remember that NO MOVES ARE HARD CODED in this engine, everything it plays is learned from the dataset.

<h3>Engine as White(Bottom) vs Me as Black(Top):</h3>

<p align="center">
  <img src="https://github.com/chun1213/Chess/blob/main/images/turn1.png" width="300" />
</p>

It plays the theoretical best move e4

It then plays the alpine opening and we reach a very humanlike game after: e5, Ne2, Nf6, Ng3, Nc6, c3,

Shown using chess.com on the right:
<p align="center">
  <img src="https://github.com/chun1213/Chess/blob/main/images/alpine.png" width="300" />
  <img src="https://github.com/chun1213/Chess/blob/main/images/alpinechess.com.png" width="300" />
</p>




<h3>Engine as Black(Top) vs Me as White(Bottom):</h3>

I play e4 as white.

<p align="center">
  <img src="https://github.com/chun1213/Chess/blob/main/images/turn1white.png" width="300" />
</p>

It plays the theoretical best move to counter e4, c5 the silican defence

This then becomes the closed silcan pin variation and we reach a very humanlike game after: c5, Nc3, e6, Nf3, f5,

Shown using chess.com on the right:
<p align="center">
  <img src="https://github.com/chun1213/Chess/blob/main/images/pinsilcan.png" width="300" />
  <img src="https://github.com/chun1213/Chess/blob/main/images/pin silican.png" width="300" />
</p>

<h1> Data </h1>

The dataset I used to train the Neural Network part of this engine was obtained from Lichess.com where a user had put together many games played between users of 2200 Elo and above (2200 Elo is very skilled). I picked this as these users would likely play the best counter moves and openings in their games. I then preprocessed the data from the pgn files and normalized the data so each board position only had one unique move that followed it. The dataset was very large but I was only able to use 10000 different games due to TensorFlow training times increasing significantly per epoch with more games. (This is with GPU CUDA acceleration, I have a GTX 1060 6GB).

See the dataset [here](https://database.nikonoel.fr/)

<h1> Architecture </h1>

The engine consists of two parts, a neural net and a minimax algorithm. The neural network was created using TensorFlow with multiple Conv2d layers with 256 filters and a 3x3 kernel size inspired by the architecture of AlphaZero. The Minimax Algorithm is exactly what it sounds like, it generates and looks through all the possible moves on a move tree and then finds the best line of moves to play from the current position. The downside is that its slow so that is why the neural net is used.

<h1> How it Works </h1>

The engine works by reading the state of the board and then inputs it into the neural network to obtain the next move.  The neural net basically memorizes popular positions and countermoves so that it can effectively play and counter any similar openings it faces, this solves the issues of most traditional engines which are bad at openings without hardcode. The minimax algorithm is there to handle very nontraditional and irregular board positions where the network predicts bad moves(move that lose material without compensation). In those cases, the minimax move is played instead.

The minimax algorithm looks 4 possible moves deep into the game from the current state of the board and outputs the best possible move. The algortihm is sped up using lapha beta pruning to allow this to be feeasable. On average, the algorithm may take anywhere between 1 second to 10 seconds to output a move based on pruning and possible moves.

<h1> Limitations </h1>

With the project explained, I need to get to the limitations of the engine. The main issue is that the neural net relies heavily on the training data I preprocessed being unique, so I ignored all other datapoints and countermoves to a board position once that position was already in the training data. A way to fix this would be to only add the move that ways played the most times as the correct answer rather than just adding the first. This would ensure that no “bad moves” would enter the engine’s training data. 

Endgames in chess are also a thing that many chess engines suffer with, as a result, most chess engines have hardcoded endgame algorithms which will take over once endgame occurs. Mine does not so the ai may suffer to see proper checkmates come endgame.

The last limitation for this AI is the shallowness of the board evaluations, the minimax algorithm works to maximize the position of the player be looking into future moves. The evaluation for these moves in my algorithm lies soley in how much a piece is worth, meaning that the algorithm does not take into account piece development, king safety or board control which are other important aspects of chess. These are things that I am looking to implement however.
