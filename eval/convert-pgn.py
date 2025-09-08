import numpy as np
import sys

# Convert an evaluation output file to a ".pgn" file which can be
# be read by the commonly used application providing ELO estimates:
#    https://www.remi-coulom.fr/Bayesian-Elo/

# Example:
#  $ model=awari_2025090117391
#  $ python eval/convert-pgn.py play-$model AOP020-02 v1000-nsim800 > play-$model-AOP020-02-v1000-800.pgn
#
# These individual .pgn files can then be concatenated and processed using something like this:
#  $ cat play-${model}-*.pgn > all-${model}.pgn
#  $ (echo readpgn all-${model}.pgn ; echo elo; echo advantage 0; echo offset 1000; echo mm 1 1;
#     echo exactdist; echo ratings; echo) | ../../../../BayesElo/bayeselo 

# prefix = "play-modelversion"
prefix = sys.argv[1]
# oracle = "AOP024-800"
oracle = sys.argv[2]
# agent = "v100"
agent = sys.argv[3]

logfile_path = prefix + "-" + agent + "-" + oracle + ".out"

fw = sys.stdout
with open(logfile_path, 'r') as f:
    try:
        iteration = 1
        round_number = 0
        line = f.readline()
        while line:
            # skip summary lines which have a space behind 'agent_win_as_first '
            if 'agent_win_as_first ' in line:
                # a summary line like:
                # agent_win_as_first 5 agent_win_as_second 7 oracle_win_as_first 0 oracle_win_as_second 0 draw 8
                line = f.readline()
                continue
            elif ('agent_win_as_first' in line) or ('agent_win_as_second' in line) or ('oracle_win_as_first' in line) or ('oracle_win_as_second' in line) or ('draw result' in line) or ('agent_turn' in line):
                # a result line ("_win_" or "draw") or
                # a line telling if the model agent started ('agent_turn 0') or
                # the oracle or external application started ('agent_turn 1').
                words = line.split()
                if words[0] == 'agent_turn':
                    if words[1] == '0':
                        agent_started = True
                        whiteplayer = agent
                        blackplayer = oracle
                    else:
                        agent_started = False
                        whiteplayer = oracle
                        blackplayer = agent
                else:
                    if words[0] == 'agent_win_as_first':
                        result = '[Result "1-0"]'
                    elif words[0] == 'agent_win_as_second':
                        result = '[Result "0-1"]'
                    elif words[0] == 'oracle_win_as_first':
                        result = '[Result "1-0"]'
                    elif words[0] == 'oracle_win_as_second':
                        result = '[Result "0-1"]'
                    else:
                        result = '[Result "1/2-1/2"]'

                    fw.write('[Event "'+ agent + "-" + oracle + " match" +'"]\n')
                    fw.write('[Iteration "'+str(iteration)+'"]\n')
                    fw.write('[Site "VU server, Amsterdam"]\n')
                    fw.write('[Round "'+str(round_number)+'"]\n')
                    fw.write('[White "'+str(whiteplayer)+'"]\n')
                    fw.write('[Black "'+str(blackplayer)+'"]\n')
                    fw.write(result + '\n')
                    fw.write('Here are detailed game moves for [Iteration "' + str(iteration)+
                             ', round' + str(round_number) + '"]\n')
                    # detailed game moves are not actually included, but also not needed for ELO
                    fw.write('\n')
                    round_number += 1
            line = f.readline()
    finally:
        f.close()
