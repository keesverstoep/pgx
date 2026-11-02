from opp import ExternalPlayer
import subprocess

# Interface to an external Gardner chess player.
# The Fairy-Stockfish application with alternative chess games
# supports this.

from pgx._src.gardner_chess_utils import TO_MAP
from pgx.gardner_chess import PAWN
gardner_map = [
        "a1", "a2", "a3", "a4", "a5",
        "b1", "b2", "b3", "b4", "b5",
        "c1", "c2", "c3", "c4", "c5",
        "d1", "d2", "d3", "d4", "d5",
        "e1", "e2", "e3", "e4", "e5",
]
gardner_mirror = [  4,  3,  2,  1,  0,
                    9,  8,  7,  6,  5,
                   14, 13, 12, 11, 10,
                   19, 18, 17, 16, 15,
                   24, 23, 22, 21, 20 ]

class GardnerOraclePlayer(ExternalPlayer):
    def __init__(self, max_depth, max_time):
        self.max_depth = max_depth
        self.max_time = max_time
        self.debug = False

    def action_convert(self, env, label, swap_color):
        from_, plane = label // 49, label % 49
        to = TO_MAP[from_, plane]
        orig_from = from_
        orig_to = to
        if swap_color:
            # mirror
            from_ = gardner_mirror[from_]
            to = gardner_mirror[to]

        if plane >= 9:
            underpromotion = -1
        else:
            underpromotion = plane // 3

        fullmove = gardner_map[from_] + gardner_map[to]

        # check promotion/underpromotion
        promote = ""
        if (env._board[0][orig_from] == PAWN) and ((orig_from % 5) == 3):
            promote = "q"
            if underpromotion == 0:
                promote = "r"
            elif underpromotion == 1:
                promote = "b"
            elif underpromotion == 2:
                promote = "n"
            fullmove = fullmove + promote

        return (from_, to, plane, underpromotion, fullmove)

    def print_valid(self, env, swap_color, trace, verbose):
        print("legal_action_mask", env.legal_action_mask[0], "swap_color", swap_color)
        valid = env.legal_action_mask[0]

        print("valid: ", end="")
        for i in range(valid.size):
            if valid[i]:
                print(" " + str(i), end="")
                label = i

                from_, to, plane, underpromotion, fullmove = self.action_convert(env, label, swap_color)
                print(" (from " + str(from_) + "=" + gardner_map[from_] +
                         " to " + str(to) + "=" + gardner_map[to] +
                         " plane " + str(plane) +
                         " up " + str(underpromotion) +
                         " fullmove " + fullmove +
                         ")",
                      end="")

    def play(self, env, swap_color, trace, verbose):
        if self.debug:
            print("play from trace", trace)

        fen_str = env._to_fen()
        if self.debug:
            print("fen:", fen_str)

        try:
            # e.g. run-stockfish rnbqk/ppppp/5/1PPP1/RNBQK w 10
            if swap_color:
                color = "b"
            else:
                color = "w"
            move = subprocess.check_output(["./eval/run-stockfish", fen_str, color, str(self.max_depth), str(self.max_time)], text=True, encoding='UTF-8')
            move = str(move).strip()
            if self.debug:
                print("stockfish result", move)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
            move = ""

        if self.debug:
            print("legal_action_mask", env.legal_action_mask[0], "swap_color", swap_color)
        valid = env.legal_action_mask[0]

        select = -1
        did_select = False
        if self.debug:
            print("valid: ", end="")

        # iterate through valid moves and check if selected move matches one
        for i in range(valid.size):
            if valid[i]:
                if self.debug:
                    print(" " + str(i), end="")
                label = i

                from_, to, plane, underpromotion, fullmove = self.action_convert(env, label, swap_color)

                if self.debug:
                    print(" (from " + str(from_) + "=" + gardner_map[from_], end="")
                    print(" to " + str(to) + "=" + gardner_map[to], end="")
                    print(" up " + str(underpromotion), end="")
                    print(" fullmove " + str(fullmove), end="")
                    print(")", end="")

                # always select one, even if we cannot find a match
                select = i
                action_cvt = fullmove

                if self.debug:
                    print("compare", action_cvt, move)
                if action_cvt == move:
                    did_select = True
                    if self.debug:
                        print('found ' + str(move))
                    break

        if not did_select:
            print('NOT found ' + str(move))
        if self.debug:
            print('op: select ' + str(select))

        return select, []

    def fen_to_board(self, fen):
        """Converts a FEN string to a 2D list representing the chessboard.
            Args: fen: The FEN string.
            Returns: A 2D list representing the chessboard.
        """
        board = []
        for row_fen in fen.split('/'):
            row = []
            for char in row_fen:
                if char == ' ':
                    # remaining non board data after this
                    break
                if char.isdigit():
                    row.extend(['  '] * int(char))
                else:
                    if char.isupper():
                        row.append(' ' + char)
                    else:
                        row.append('b' + char)
            board.append(row)
        return board

    def print_oracle_status(self, env, trace, swap_color):
        # TODO print actual oracle status, now only done verbose above in play()
        # observe = env.observation
        # print("observe", observe)
        print("move trace", trace)
        print("board", env._board)
        fen_str = env._to_fen()
        print("fen:", fen_str)

        board = self.fen_to_board(fen_str)
        for row in board:
            print(row)
        return None
