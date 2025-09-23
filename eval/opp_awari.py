from opp import ExternalPlayer
import random

# Awari Oracle database support

# The awari_db/ directory contains a slightly extended version
# of the code providing access to the Awari Oracle score database,
# which can be downloaded from here:
#   https://data.yoda.vu.nl:9443/vault-beta-cs-awari-retrograde-analysis/awari-score-database-v1.0%5B1711496941%5D/original/

from awari_db.db_board import db_board
from awari_db.db_lookup import gen_board_and_child_scores

next = [
    [ 1, 2, 3, 4, 5, 7, 7, 8,  9, 10, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 8, 8,  9, 10, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 7, 9,  9, 10, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8,  9, 11, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8,  9, 10,  0, 0, ],
    # 0  1  2  3  4  5  6  7   8   9  10  11 
]

class AwariOraclePlayer(ExternalPlayer):
    def __init__(self, mistake_fraction = 0.0, mistake_max = 4):
        self.mistake_fraction = mistake_fraction
        self.mistake_max = mistake_max
        # enable next if additional database is available
        self.quick_db = True

    def print_valid(self, env, swap_color, trace, verbose):
        print("legal_action_mask", env.legal_action_mask[0], "swap_color", swap_color)
        valid = env.legal_action_mask[0]

        print("valid: ", end="")
        for i in range(valid.size):
            if valid[i]:
                print(" " + str(i), end="")
        print("")
        
    def can_move(self, pieces, player):
        if player == 1:
            for i in range(6):
                if pieces[i] > 0:
                    return True
        else:
            for i in range(6, 12):
                if pieces[i] > 0:
                    return True
        return False

    def sow(self, pieces, move, player, verbose):
        pit1 = move
        seeds = pieces[pit1]
        pit2 = pit1 + 6
        capture = 0
        oldseeds = seeds
        next_ptr = next[pit1]

        child_pieces = pieces[6:12] + pieces[0:6] + pieces[18:24] + pieces[12:18]

        # take away seeds
        child_pieces[pit2] = 0

        # sow them
        while seeds > 0:
            pit2 = next_ptr[pit2]
            child_pieces[pit2] += 1
            seeds -= 1

        captures = 0
        while (pit2 < 6 and pit2 >= 0) and ((child_pieces[pit2] == 2) or (child_pieces[pit2] == 3)):
            # capture
            captures += child_pieces[pit2]
            child_pieces[pit2] = 0
            pit2 -= 1

        # now return the board in the canonical player=1  orientation
        ret_child_pieces = child_pieces[6:12] + child_pieces[0:6] + child_pieces[18:24] + child_pieces[12:18]

        # update the captures in the returned child
        if captures > 0:
            if verbose: print('captured ' + str(captures) + ' by ' + str(player))

            ret_child_pieces[17] += captures

        return ret_child_pieces
    
    def get_legal_moves(self, pieces, verbose):
        """Returns all the legal moves for the current player, assuming canonical board
        """
        moves = set()  # stores the legal moves.

        # canonical board, so player is 1, not -1
        player = 1 
        if verbose:
            print('get legal moves of player ', player, ' for ', pieces)

        # first find moves that allow the opponent to respond
        # NOTE: board is in canonical form
        for i in range(6):
            if pieces[i] != 0:
                # check that when doing this move, the opponent can respond
                board = self.sow(pieces, i, player, verbose)
                if self.can_move(board, - player):
                    if verbose: print('add move ', i, 'since opponent can move in', board)
                    moves.add(i)
                else:
                    if verbose: print('skip move ', i, 'since opponent cannot move in', board)

        if len(moves) == 0:
            # retry, now adding moves that will allow no response
            for i in range(6):
                if pieces[i] != 0:
                    if verbose: print('add move ' + str(i))
                    moves.add(i)

        return list(moves)
    
    def valid_moves(self, pieces, player, verbose):
        # return a fixed size binary vector
        valids = [0] * 7  ## self.action_size()

        all_pieces = pieces + [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0]
        legalMoves = self.get_legal_moves(all_pieces, verbose)
        if len(legalMoves) == 0:
            valids[-1] = 1
        else:
            for x in legalMoves:
                valids[x] = 1

        return valids

    def play(self, env, swap_color, trace, verbose):
        observe = env.observation
        obs = observe[0,0]

        #print("play from observation", obs)

        pieces = self.obs_to_pieces(observe)
        if verbose:
            print("pieces", pieces)

        # canoncial board is as seen from player to move:
        player = 1 

        # neede to do a validity check since the reported scores may
        # not be consistent with this, as they give a max score
        # that may dependent on wheher a wipeout is acceptable or not
        valid = self.valid_moves(pieces, player, verbose)
        if verbose:
            print('op: possible moves: ', valid)

        board = db_board(pieces[0], pieces[1], pieces[2], pieces[3], pieces[4], pieces[5],
                         pieces[6], pieces[7], pieces[8], pieces[9], pieces[10], pieces[11])

        overall_score, scores, _ = gen_board_and_child_scores(board)
        if verbose:
            print("score", overall_score)
            print("child_scores", scores)

        preferred = [ True, True, True, True, True, True ]
        if self.quick_db:
            # For positions up to 21 stones there may be a preference for moves that
            # lead to a quicker win, e.g., ones that may avoid a draw to repeated position.
            # They are bitmaps stored in a separtely constructed "quick" database.
            seeds = board.count_seeds()
            if seeds < 21:
                g = board.GoedelNumber(seeds)
                quick_score = board.quick_score(g, seeds)
                if verbose:
                    print("quick_score", quick_score)
                if quick_score != 0:
                    if (quick_score & 0x01) == 0:
                        preferred[0] = False
                    if (quick_score & 0x02) == 0:
                        preferred[1] = False
                    if (quick_score & 0x04) == 0:
                        preferred[2] = False
                    if (quick_score & 0x08) == 0:
                        preferred[3] = False
                    if (quick_score & 0x10) == 0:
                        preferred[4] = False
                    if (quick_score & 0x20) == 0:
                        preferred[5] = False

        best = -127
        best_i = 6
        # first find the best score, picking an arbitrary best when the same
        for i in range(6):
            # NOTE: the validity may not be observed by the scores
            if valid[i] and preferred[i] and  scores[i] != 127:
                # pick best move, or choose between best ones;
                # allow for mistakes if so desired
                if scores[i] > best or (scores[i] == best and random.random() < 0.5):
                    best = scores[i]
                    best_i = i

        # find the next best score, picking an arbitrary next best when the same
        next_best = -127
        next_best_i = 6
        for i in range(6):
            # NOTE: the validity may not be observed by the scores
            if valid[i] and scores[i] != 127:
                # pick best move, or choose between best ones;
                # allow for mistakes if so desired
                if (scores[i] < best and scores[i] > next_best) or (scores[i] == next_best and random.random() < 0.5):
                    next_best = scores[i]
                    next_best_i = i

        select = best_i
        if random.random() < self.mistake_fraction:
            # take other choice, if not absurd:
            if next_best != -127 and (best - next_best) <= self.mistake_max:
                select = next_best_i
                if verbose:
                    print('op: select suboptimal move')

        if verbose:
            print('op: select ' + str(select))

        return select

    def evaluate(self, env, action, verbose):
        observe = env.observation

        pieces = self.obs_to_pieces(observe)
        if verbose:
            print("pieces", pieces)

        # canoncial board is as seen from player to move:
        player = 1 

        # neede to do a validity check since the reported scores may
        # not be consistent with this, as they give a max score
        # that may dependent on wheher a wipeout is acceptable or not
        valid = self.valid_moves(pieces, player, verbose)
        if verbose:
            print('op: possible moves: ', valid)

        b = db_board(pieces[0], pieces[1], pieces[2], pieces[3], pieces[4], pieces[5],
                     pieces[6], pieces[7], pieces[8], pieces[9], pieces[10], pieces[11])

        overall_score, scores, _ = gen_board_and_child_scores(b)
        if verbose:
            print("score", overall_score)
            print("child_scores", scores)

        best = -127
        # find the best score, it's arbitrary when two moves score are the same
        for i in range(6):
            # NOTE: the validity may not be observed by the scores
            if valid[i] and scores[i] != 127:
                if scores[i] > best:
                    best = scores[i]

        if best == -127 or action == 6:
            # pass
            return 0

        if verbose:
            # print the best actions for comparison
            best_actions = []
            for i in range(6):
                if valid[i] and scores[i] != 127:
                    if scores[i] == best:
                        best_actions.append(i)
            print("best actions", best_actions, "score", best, "selected", action, "score", scores[action])

        return (scores[action] - best)

    def obs_to_pieces(self, observe):
        pieces = [ int(observe[0,0,0,0]), int(observe[0,0,1,0]), int(observe[0,0,2,0]), int(observe[0,0,3,0]),
                   int(observe[0,0,4,0]), int(observe[0,0,5,0]), int(observe[0,1,0,1]), int(observe[0,1,1,1]),
                   int(observe[0,1,2,1]), int(observe[0,1,3,1]), int(observe[0,1,4,1]), int(observe[0,1,5,1]) ]
        # print("pieces", pieces)
        return pieces

    def print_oracle_status(self, env, trace, swap_color):
        # use Python interface for the published Awari database dataset

        # get a canonical view, seen from the current player
        observe = env.observation
        pieces = self.obs_to_pieces(observe)
        print("pieces", pieces)

        b = db_board(int(pieces[0]), int(pieces[1]), int(pieces[2]),
                              int(pieces[3]), int(pieces[4]), int(pieces[5]),
                              int(pieces[6]), int(pieces[7]), int(pieces[8]),
                              int(pieces[9]), int(pieces[10]), int(pieces[11]))
        score, child_scores, _ = gen_board_and_child_scores(b)
        print("score", score)
        print("child_scores", child_scores)
        return child_scores
