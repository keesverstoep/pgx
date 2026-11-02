from opp import ExternalPlayer
import random
import ctypes

# See https://github.com/PascalPons/connect4
# with Python interface implemented in
#    https://github.com/TonyCongqianWang/connect4_solver_fork
# Code below is based on c_interface_test_linux.py of the second link,
# assumes the connect4 shared library is preconfigured, and that it
# can access the required 7x6.book database.

class Connect4OraclePlayer(ExternalPlayer):
    def __init__(self, mistake_fraction, mistake_max):
        # Load the shared library, for now assumes it is in the run directory
        solver_lib = ctypes.CDLL(r"./c4solver_c_interface.so")

        # Define argument and return types
        solver_lib.solver_init.argtypes = [ctypes.c_char_p]
        solver_lib.solver_init.restype = ctypes.POINTER(ctypes.c_void_p)
        solver_lib.solver_delete.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        solver_lib.solver_delete.restype = None
        solver_lib.solver_solve.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_char_p, ctypes.c_size_t]
        solver_lib.solver_solve.restype = ctypes.c_char_p

        self.solver_lib = solver_lib
        self.handle = self.solver_lib.solver_init(None) 
        self.result_buffer = ctypes.create_string_buffer(256)

        self.mistake_fraction = mistake_fraction
        self.mistake_max = mistake_max
        self.debug = False

    def print_valid(self, env, swap_color, trace, verbose):
        print("legal_action_mask", env.legal_action_mask[0], "swap_color", swap_color)
        valid = env.legal_action_mask[0]

        print("valid: ", end="")
        for i in range(valid.size):
            if valid[i]:
                print(" " + str(i), end="")
        print("")
        
    def play(self, env, swap_color, trace, verbose):
        if self.debug:
            print("play from trace", trace)

        tracebuf = ""
        # moves/columns are numbered from 1
        for move in trace:
            tracebuf += chr(ord("1") + move)
        if self.debug:
            print("tracebuf", tracebuf)

        bytebuf = tracebuf.encode("utf-8")
        # print("bytebuf", bytebuf)

        result = self.solver_lib.solver_solve(self.handle, bytebuf, False, True, self.result_buffer, 256)
        str_result = result.decode()
        # print("result", str_result)
        result_array = str_result.split()
        # print("result_array", result_array)
        int_array = [int(numeric_string) for numeric_string in result_array]
        # print("int_array", int_array)
        # only keep last 7 entries, i.e., remove optional prefix
        scores = int_array[-7:]
        if verbose:
            print("scores", scores)

        if self.debug:
            print("legal_action_mask", env.legal_action_mask)
        valid = env.legal_action_mask[0]

        # Find best move
        best = -127
        best_i = 7
        # first find the best score, picking an arbitrary best when the same
        for i in range(7):
            # NOTE: the validity may not be observed by the scores
            if valid[i] and abs(scores[i]) < 42:  ## TODO sanity check
                # pick best move, or choose between best ones;
                # allow for mistakes if so desired
                if scores[i] > best or (scores[i] == best and random.random() < 0.5):
                    best = scores[i]
                    best_i = i

        # find the next best score, picking an arbitrary next best when the same
        next_best = -127
        next_best_i = 7
        for i in range(7):
            # NOTE: the validity may not be observed by the scores
            if valid[i] and abs(scores[i]) < 42:
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
        if self.debug:
            print('op: select ' + str(select))

        return select, scores

    def print_oracle_status(self, env, trace, swap_color):
        # TODO print actual oracle status, now only done verbose above in play()
        # observe = env.observation
        # print("observe", observe)
        print("state trace", trace)
        return None
