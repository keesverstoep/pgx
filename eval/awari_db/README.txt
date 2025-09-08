Awari score database

This directory contains a basic Python implementation providing access
to the score database of the Awari game, discussed in the publication:

John Romein, Henri E. Bal (2003):
"Solving the Game of Awari using Parallel Retrograde Analysis",
IEEE Computer, Vol. 36, No. 10

The location of the Awari score database (consisting of files
scores.[0-46] and scores.48) can be configured using environment
variable "AWARI_PATH".  If this is not set, the database files
are assumed to be in the current directory.

The arguments of the db_lookup.py tool are the number of
seeds in the Awari pits in counter-clockwise order, starting
with the leftmost pit of the player whose turn it is.

Example usage:
======
$ export AWARI_PATH=../database
$ python db_lookup.py 1 1 1 1 1 2 4 2 3 0 0 0
board: 1 1 1 1 1 2 4 2 3 0 0 0 
score: 0
children scores:  -1 -1 -2 -2 -2 0 
======

The interpretation of the output is as follows:
- the "board" line repeats the input arguments after parsing
  (mostly to mimic the interface of the original tool written in C,
  which has been used in some existing machine learning projects));
- the "score" line prints the value of the current position;
- the "children scores" line prints the value after doing
  each of the 6 potential moves
  - special value 127 means a particular move is not available.

The score printed is the score of the current board interpreted
according to the special rules regarding which moves are acceptable.
Specifically, it is not allowed to remove all stones of the opponent
(leaving it no move), unless it is the only move available.

For a more direct interface to the database scores, giving just the
plain score computed for a position, use the interface from Board.py
as follows:
======
    b =  Board(pits[0],pits[1],pits[2],pits[3],pits[4],pits[5],
               pits[6],pits[7],pits[8],pits[9],pits[10],pits[11])
    seeds = b.count_seeds()
    g = b.GoedelNumber(seeds) 
    score = b.score(g, seeds)
======
For more examples, see file Board.py, and run the tests contained
in it using "python Board.py"

NOTE: This implementation assumes the database coding that omits
unreachable positions, i.e., the opponent always has at least one
empty pit (except for the starting position), since in the previous
turn the opponent must have emptied a pit.
