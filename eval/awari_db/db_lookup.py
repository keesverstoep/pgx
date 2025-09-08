#!/usr/bin/env python3

#
# Copyright 2024 Vrije Universiteit Amsterdam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This code is patterned after the original db_lookup tool printing
# a position state and the child position score states.

import sys
# import Board
from . import db_board

NO_MOVE	= 127

def board_mirror(src):
    dst = [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0]
    for i in range(6):
       dst[i] = src[6 + i]
    for i in range(6):
       dst[6 + i] = src[i]
    return dst

def can_do_move(pits):
    for i in range(6):
        if pits[i] != 0:
            return True
    return False

def gen_board_and_child_scores(board):
    capture_seeds = [
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 
    ]

    next = [
        [ 1, 2, 3, 4, 5, 7, 7, 8,  9, 10, 11, 0, ],
        [ 1, 2, 3, 4, 5, 6, 8, 8,  9, 10, 11, 0, ],
        [ 1, 2, 3, 4, 5, 6, 7, 9,  9, 10, 11, 0, ],
        [ 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 0, ],
        [ 1, 2, 3, 4, 5, 6, 7, 8,  9, 11, 11, 0, ],
        [ 1, 2, 3, 4, 5, 6, 7, 8,  9, 10,  0, 0, ],
    ]

    children_scores = [ NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE]

    nr_seeds = board.count_seeds()
    has_eradicated_children = False
    has_children = False
    reply_score = - nr_seeds

    if nr_seeds == 47 or nr_seeds > 48:
        print("impossible number of seeds", nr_seeds)
        return None, None

    # generate up to 6 child boards according to awari rules and
    # update individual and resulting scores accordingly
    for pit1 in range(5, -1, -1):
        seeds = board.pits[pit1]
        if seeds != 0:
            child_pits = board_mirror(board.pits)
            pit2 = pit1 + 6
            child_pits[pit2] = 0
            next_ptr = next[pit1]

            while seeds > 0:
                pit2 = next_ptr[pit2]
                child_pits[pit2] += 1
                seeds -= 1

            while pit2 >= 0 and pit2 < 6 and capture_seeds[child_pits [pit2]] == 1:
                child_pits[pit2] = 0   # capture
                pit2 -= 1

            if not can_do_move(child_pits):
                if (not has_eradicated_children) and has_children:
                    continue

                has_eradicated_children = True
            elif has_eradicated_children:
                reply_score = -nr_seeds
                has_eradicated_children = False

            has_children = True

            # child = Board.Board(child_pits[0], child_pits[1], child_pits[2],
            child = db_board.db_board(child_pits[0], child_pits[1], child_pits[2],
                          child_pits[3], child_pits[4], child_pits[5],
                          child_pits[6], child_pits[7], child_pits[8],
                          child_pits[9], child_pits[10], child_pits[11])
            child_seeds = child.count_seeds()
            child_goedel = child.GoedelNumber(child_seeds)
            lookup_score = child.score(child_goedel, child_seeds)
            children_scores[pit1] = nr_seeds - child_seeds - lookup_score

            if reply_score < children_scores[pit1]:
                reply_score = children_scores[pit1]

    return reply_score, children_scores


if __name__ == "__main__":
    if len(sys.argv) != 13:
        print("Usage: %s [pit1] .. [pit12]" % sys.argv[0])
        sys.exit(1)

    # b = Board.Board(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
    b = db_board.db_board(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
                    int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
                    int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]),
                    int(sys.argv[10]), int(sys.argv[11]), int(sys.argv[12]))

    print("board:", end=" ")
    for i in range(12):
        print(b.pits[i], end=" ")
    print()

    score, child_scores = gen_board_and_child_scores(b)
    print("score:", score)
    print("children scores: ", end=" ")
    for i in range(len(child_scores)):
         print(str(child_scores[i]), end=" ")
    print()

    # no "quick moves" database interface yet
