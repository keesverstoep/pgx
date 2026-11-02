from opp import ExternalPlayer

class OthelloOraclePlayer(ExternalPlayer):
    def __init__(self, max_depth, max_time):
        self.max_depth = max_depth
        self.max_time = max_time
        self.debug = False

    def board_to_fen(self, state, swap_color):
        """Converts the current board to pseudo FEN notation suitable for the oracle
            Args: self state
            Returns: FEN string representing the board.

            Example:
            --------/--------/--------/---bwb--/---wbb--/--------/--------/--------
        """
        fen = ""
        n = 0
        for val in list(state._x.board[0]):
            if val == 1:
                if swap_color:
                    fen += "w"
                else:
                    fen += "b"
            elif val == -1:
                if swap_color:
                    fen += "b"
                else:
                    fen += "w"
            else:
                fen += "-"
            n += 1
            if n % 8 == 0:
                fen += "/"

        return fen

    def print_valid(self, env, swap_color, trace, verbose):
        if self.debug:
            print("legal_action_mask", env.legal_action_mask[0], "swap_color", swap_color)
        valid = env.legal_action_mask[0]

        print("valid: ", end="")
        for i in range(valid.size):
            if valid[i]:
                print(" " + str(i), end="")
        print("")

    def play(self, state, swap_color, trace, verbose):
        # print("play from trace", trace)

        # create fen string of board
        fen_str = self.board_to_fen(state, swap_color)
        if self.debug:
            print("fen:", fen_str)

        import subprocess
        try:
            # E.g., run-edax --------/--------/--------/---bwb--/---wbb--/--------/--------/-------- w maxdepth time
            if swap_color:
                # color = "b"
                color = "w"
            else:
                # color = "w"
                color = "b"
            if self.debug:
                print("check oracle fen", fen_str, "color", color)
            move_result = subprocess.check_output(["./eval/run-edax", fen_str, color, str(self.max_depth), str(self.max_time)], text=True, encoding='UTF-8')
            move_result = str(move_result).strip()
            if self.debug:
                print("edax result", move_result)
            # convert board cell ID of the form <LETTER><number> to numeric move ID
            if len(move_result) >= 2:
                move = (ord(move_result[1]) - ord('1')) * 8 + (ord(move_result[0]) - ord('A'))
            else:
                # pass
                move = 64
            if self.debug:
                print("numeric:", move)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
            move = -1

        # print("state", state)
        if self.debug:
            print("legal_action_mask", state.legal_action_mask[0], "swap_color", swap_color)
        valid = state.legal_action_mask[0]

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

                # always select one, even if we cannot find a match
                select = i
                action_cvt = i

                if self.debug:
                    print(" compare", action_cvt, move)
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

    def print_oracle_status(self, state, trace, swap_color):
        # TODO print actual oracle status, now only done verbose above in play()
        print("state trace", trace)
        print("board", state._x.board)
        fen_str = self.board_to_fen(state, swap_color)

        print("fen:", fen_str)
        print("board:", end="")
        for c in fen_str:
            if c == "/":
                print("", end="")
            else:
                print(c, end="")
        print()
        return None
