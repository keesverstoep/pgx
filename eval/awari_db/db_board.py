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

import os
import goedels
import binomium

awari_path = os.getenv("AWARI_PATH", ".") 

# Based on the original version written in C.
#
# This implements the coding used that omits unreachable positions,
# i.e., the opponent always has at least one empty pit (except for
# the starting position) since it in the previous turn it must
# have emptied one

# Expand the goedel array to full 12,64,64] dimensions
def full_goedels():
    goedels_expanded = [[[0 for z in range(64)] for y in range(64)] for x in range(12)]
    for x in range(len(goedels.goedels)):
        for y in range(len(goedels.goedels[x])):
            for z in range(len(goedels.goedels[x][y])):
                goedels_expanded[x][y][z] = goedels.goedels[x][y][z]
    return goedels_expanded 
    
goedel_arr = full_goedels()
goedel_state_counts = []
goedel_cumulative_state_counts = []


class db_board():
    def __init__(self, pit0, pit1, pit2, pit3, pit4, pit5, pit6, pit7, pit8, pit9, pit10, pit11):
        self.pits = [pit0, pit1, pit2, pit3, pit4, pit5, pit6, pit7, pit8, pit9, pit10, pit11]

    @classmethod
    def goedel_init(cls, goedelNumber, nrSeeds):
        pits = [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0]
        pit = 11;
        index = nrSeeds - 5

        while True:
            pits[pit] = 0
            while nrSeeds > 0:
                newGoedel = goedelNumber - binomium.binomium[pit][nrSeeds + 8]
                if pits[pit] > 0:
                    newGoedel += binomium.binomium[pit][index + 8]
                if newGoedel < 0:
                    break
                goedelNumber = newGoedel

                nrSeeds -= 1
                pits[pit] += 1
                index -= 1

            index += 1
            if pits[pit] == 0:
                pit -= 1
                break
            pit -= 1

        while True:
            seeds = 0
            goedelsPtr = goedel_arr[pit][nrSeeds + 8]

            while nrSeeds > seeds and goedelNumber >= goedelsPtr[seeds + 1]:
                seeds += 1

            goedelNumber -= goedelsPtr[seeds]
            nrSeeds -= seeds
            pits[pit] = seeds
            pit -= 1
            if pit < 0:
                break

        return cls(pits[0], pits[1], pits[2], pits[3], pits[4], pits[5],
                   pits[6], pits[7], pits[8], pits[9], pits[10], pits[11])

    def count_seeds(self):
        sum = 0
        for i in range(12):
            sum += self.pits[i]
        return sum

    def GoedelNumber(self, nrSeeds):
        goedelNumber = 0
        seeds = 0
        pit = 11

        # NOTE: this code makes use of the fact that the opponent (except from the
        # starting position) always has at least one pit empty because of a previous move
        while self.pits[pit] > 0:
            seeds = self.pits[pit]
            goedelNumber += goedel_arr[pit][nrSeeds + 8][seeds]
            goedelNumber -= goedel_arr[pit][nrSeeds - pit + 5 + 8][seeds - 1]
            nrSeeds -= seeds;
            pit -= 1
            if pit < 0:
                # should just happen for the starting position, TODO: check
                print("GoedelNumber should only be called for positions where opponent has moved")
                return -1

        while nrSeeds > 0:
            seeds = self.pits[pit]
            goedelNumber += goedel_arr[pit][nrSeeds + 8][seeds]
            pit -= 1
            nrSeeds -= seeds

        return goedelNumber

    def nrStates(self, nrSeeds):
        return goedel_arr[11][nrSeeds + 8][6]

    def next_board(self):
        pits = self.pits.copy()
        if pits[0] > 0:
            pits[0] -= 1
            pits[1] += 1
        else:
            # pits[12] = 1; /* stop next loop at end of pits[] */
            pit = 0;

            while pit < 12 and pits[pit] == 0:
                pit += 1

            if pit < 12:
                pits[0] = pits[pit] - 1
                pits[pit] = 0
                if pit < 11:
                    pits[pit + 1] += 1

            #if defined OMIT_UNREACHABLE_STATES
            if pit == 5 and (pits[6] > 0 and pits[7] > 0 and pits[8] > 0 and 
                            pits[9] > 0 and pits[10] > 0 and pits[11] > 0):
                pits[6] = 0
                pits[7] += 1
        return db_board(pits[0],pits[1],pits[2],pits[3],pits[4],pits[5],
	                pits[6],pits[7],pits[8],pits[9],pits[10],pits[11])

    def quick_score(self, goedelNumber, nrSeeds):
        if nrSeeds >= 21:
            return 0

        if nrSeeds == 0:
            cumulative = 0
        else:
            cumulative = goedel_cumulative_state_counts[nrSeeds - 1]
        qdb_index = cumulative + goedelNumber

        # file has a compact list of 6-bit bitmaps per position, so need to pull them out
        # every chunk of 3 bytes i.e. 24 bits contains 4 values,
        shift =  [0, 6, 4, 2]
        compact_index = (3 * qdb_index) >> 2
        # print("compact_index", compact_index)
        filename = awari_path + "/awari.qdb"
        try:
            f = open(filename, 'rb')
            f.seek(compact_index)
            bytes = f.read(4)
            f.close()

            result1 = int.from_bytes(bytes, byteorder="little", signed=True)
            result2 = result1 >> shift[qdb_index & 3]
            result3 = result2 & 0x3F;
            return result3
        except Exception as e:
            # fail silently: the quick db may not be available but is not essential
            return 0
    
    def score(self, goedelNumber, nrSeeds):
        filename = awari_path + "/scores.%d" % nrSeeds
        f = open(filename, 'rb')
        f.seek(goedelNumber)
        byte = f.read(1)
        f.close()
        result = int.from_bytes(byte, byteorder="little", signed=True) 
        return result

    def __repr__(self):
        str = "+----+----+----+----+----+----+\n|"

        pit = 12
        while pit > 6:
            pit -= 1
            str += "%3d |" % self.pits[pit]

        str += "\n+----+----+----+----+----+----+\n|"

        pit = 0
        while pit < 6:
            str += "%3d |" % self.pits[pit]
            pit += 1

        str += "\n+----+----+----+----+----+----+"
        return str

def goedel_set_state_counts():
    global goedel_state_counts
    global goedel_cumulative_state_counts

    total = 0
    for i in range(48):
        if i == 47:
            # 47 stone db itself does not exist
            goedel_state_counts.append(0)
            # copy previous cumulative count
            goedel_cumulative_state_counts.append(goedel_cumulative_state_counts[-1])
            continue
        b =  db_board(i, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0)
        nrstates = b.nrStates(i)
        total += nrstates
        # print("seeds", i, "nrstates", nrstates, "total", total)
        goedel_state_counts.append(nrstates)
        goedel_cumulative_state_counts.append(total)
    # print("goedel_state_counts", goedel_state_counts)
    # print("goedel_cumulative_state_counts", goedel_cumulative_state_counts)

# init:
goedel_set_state_counts()

def get_goedel_cumulative_state_count(nr_seeds):
    return goedel_cumulative_state_counts[nr_seeds]

def test_board(pits):
    b =  db_board(pits[0],pits[1],pits[2],pits[3],pits[4],pits[5],
	       pits[6],pits[7],pits[8],pits[9],pits[10],pits[11])
    print(b)
    seeds = b.count_seeds()
    g = b.GoedelNumber(seeds)
    print("seeds", seeds, "goedel", g)

    new_b = Board.goedel_init(g, seeds)
    print("new board:\n" + str(new_b))
    print()

def test_position(pits):
    b =  db_board(pits[0],pits[1],pits[2],pits[3],pits[4],pits[5],
	       pits[6],pits[7],pits[8],pits[9],pits[10],pits[11])
    print(b)
    seeds = b.count_seeds()
    g = b.GoedelNumber(seeds)
    score = b.score(g, seeds)
    print("nrStates", b.nrStates(seeds)) 
    print("score:", score)
    quick_score = b.quick_score(g, seeds)
    print("quick:", quick_score)

if __name__ == "__main__":
    # test_board([1,0,0,0,0,0,  0,0,0,0,0,0])
    test_position([0,1,1,0,0,0,  0,0,0,0,0,0])
