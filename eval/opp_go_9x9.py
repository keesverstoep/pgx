from opp import ExternalPlayer

class GoOraclePlayer(ExternalPlayer):
    def __init__(self, max_depth, max_time):
        self.max_depth = max_depth
        self.max_time = max_time
        self.debug = False

    def print_valid(self, env, swap_color, trace, verbose):
        # print("play from trace", trace)

        # print("env", env)
        print("legal_action_mask", env.legal_action_mask[0], "swap_color", swap_color)
        valid = env.legal_action_mask[0]

        print("valid: ", end="")
        for i in range(valid.size):
            if valid[i]:
                print(" " + str(i), end="")
        print("")

    def play(self, state, swap_color, trace, verbose):
        if self.debug:
            print("play from trace", trace)

        import subprocess
        try:
            if swap_color:
                # color = "b"
                color = "w"
            else:
                # color = "w"
                color = "b"
            if self.debug:
                print("check oracle ", "color", color)
            # run external Go player for one move from current position
            # cmdlist = ["./eval/run-pachi", str(self.max_depth), str(self.max_time), color]
            cmdlist = ["./eval/run-katago", str(self.max_depth), str(self.max_time), color]
            # add the move trace so far
            for move in trace:
                if move >= 0 and move < 81:
                    move_col = move % 9
                    move_row = move // 9
                    move_char = chr(ord("A") + move_col)
                    # exception:
                    if move_char == "I":
                        move_char = "J"
                    move_cvt = move_char + chr(ord("1") + move_row)
                else:
                    move_cvt = "PASS"
                cmdlist.append(move_cvt)
            if self.debug:
                print("command with converted trace:", cmdlist)
            # default: pass
            move = 81
            for attempt in range(3):
                # something going wrong occasionally???
                move_result = subprocess.check_output(cmdlist, text=True, encoding='UTF-8')
                move_result = str(move_result).strip()
                if self.debug:
                    print("command result", move_result)
                # convert Pachi's board cell ID of the form <LETTER><number> back to numeric move ID
                # in the model's encoding:
                if len(move_result) == 2:
                    move_row = ord(move_result[1]) - ord('1')
                    move_char = move_result[0]
                    # exception:
                    if move_char == "J":
                        move_char = "I"
                    move = move_row * 9 + (ord(move_char) - ord('A'))
                    break
                elif "PASS" in move_result or "pass" in move_result:
                    break
                else:
                    # something wrong? Retry
                    if self.debug:
                        print("command retry", attempt)
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

        return select

    def print_oracle_status(self, state, trace, swap_color):
        # TODO print actual oracle status, now only done verbose above in play()
        print("move trace", trace)
        print("board", state._x.board)
        return None
