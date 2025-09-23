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

# Based on the original version written in C.

import os
import goedels
import binomium
import db_board
import db_lookup

awari_path = os.getenv("AWARI_PATH", ".") 

class state_info:
    def __init__(self, id, board, seeds, score):
        self.id = id
        self.board = board
        self.seeds = seeds
        self.score = score
        self.depth = 0
        self.plausible_moves = 0
        self.good_moves = 0

all_states = []

# For debugging:
verbose = False

def init_states(nr_seeds, tot_nr_states):
    # Init from regular database
    # For seed counts up to e.g. 21 this should fit in memory
    state_id = 0
    for seeds in range(nr_seeds + 1):
        board = db_board.db_board(seeds, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        nr_states = board.nrStates(seeds)
        if verbose:
            print("nr states for seeds", seeds, "=", nr_states)
        for j in range(nr_states):
            if verbose:
                print("index", state_id)
                print(board)
            g = board.GoedelNumber(seeds)
            score = board.score(g, seeds)
            if verbose:
                print("goedel", g, "score", score)
            newstate = state_info(state_id, board, seeds, score)
            all_states.append(newstate)
            state_id += 1
            board = board.next_board()
    for i in range(state_id):
        if verbose:
            print("analyzing state", i)
        state = all_states[i]
        board = state.board
        _, _, children = db_lookup.gen_board_and_child_scores(board)
        if verbose:
            print("children", children)
        if children[0] < 0 and children[1] < 0 and children[2] < 0 and \
           children[3] < 0 and children[4] < 0 and children[5] < 0:
            state.depth = 0
        else:
            state.depth = 0xFF
        for move in range(6):
            child_id = children[move]
            if child_id >= 0:
                child = all_states[child_id]
                if state.score == state.seeds - child.seeds - child.score:
                    if verbose:
                        print("move", move, "child", child_id, "plausible")
                    state.plausible_moves |= (1 << move)
                    if verbose:
                        print("init state", i, "plausible update", state.plausible_moves)
                else:
                    if verbose:
                        print("move", move, "child", child_id, "implausible")
        if verbose:
            print("state", i, "plausible", state.plausible_moves)


def do_work(nr_entries):
    changed_something = True

    depth = 1
    while changed_something:
        print("depth =", depth);
        changed_something = False

        for i in range(nr_entries):
            if all_states[i].depth == 0xFF:
                state = all_states[i]
                _,_, children = db_lookup.gen_board_and_child_scores(state.board)

                if verbose:
                    print("work on", i)
                for j in range(6):
                    if state.plausible_moves & (1 << j):
                        child = children[j]
                        child_depth = all_states[child].depth
                        capture = (state.seeds > all_states[child].seeds)
                        if verbose:
                            print("plausible", j, "child", child, "capture", capture)

                        if (not capture) and (state.score < 0) and (child_depth == 0xFF):
                            if verbose:
                                print("no capture; score", state.score, "child_depth", child_depth)
                            state.good_moves = 0
                            state.depth      = 0xFF
                            break

                        if capture or child_depth == depth - 1:
                            if verbose:
                                print("capture; child_depth ", child_depth)
                            state.good_moves |= (1 << j)
                            state.depth = depth
                            changed_something = True
        depth += 1

    for i in range(nr_entries):
        state = all_states[i]
        if verbose:
            print("board", i)
            print(state.board)
            print("index", i, "score", state.score, "depth", state.depth, "plausible",
                  state.plausible_moves, "good_moves", state.good_moves)

        _,_, children = db_lookup.gen_board_and_child_scores(state.board)
        for j in range(6):
            if children[j] >= 0:
                if verbose:
                    print("child", j, "index ", children[j])
                child = all_states[children[j]]
                if verbose:
                    print(child.board)
                    print("  child index ", children[j], "score", child.score,
                      "depth =", child.depth, "plausible_moves =", child.plausible_moves,
                      "good_moves", child.good_moves)

    
def write_results(nr_entries):
    filename = awari_path + "/awari.qdb.new"
    try:
        f = open(filename, 'wb')
        val = 0
        for i in range(nr_entries):
            state = all_states[i]

            # file has a compact list of 6-bit bitmaps per position, so need to write
            # them as chunks of 3 bytes i.e. 24 bits contains 4 values,
            shift =  [0, 6, 12, 18]
            if i & 3 == 0:
                if i > 0:
                    arr =  val.to_bytes(3, 'little')
                    f.write(arr)
                    if verbose:
                        print("write", i, "val", val)
                val = 0

            good_moves = state.good_moves
            if verbose:
                print("value", i, ":", good_moves)
            if good_moves == 0:
                good_moves = state.plausible_moves
                if verbose:
                    print("new value", i, ":", good_moves)
            val |= (good_moves << shift [i & 3])
        # flush any last states
        arr =  val.to_bytes(3, 'little')
        f.write(arr)
        f.close()
    except Exception as e:
        print("Error writing database", filename, e)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: %s <seeds>" % sys.argv[0])
        sys.exit(1)

    nr_seeds = int(sys.argv[1])
    nr_entries = db_board.get_goedel_cumulative_state_count(nr_seeds)
    print("nr_entries =", nr_entries)
    init_states(nr_seeds, nr_entries)
    do_work(nr_entries)   
    write_results(nr_entries)
