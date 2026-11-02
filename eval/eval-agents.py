# Copyright 2025 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluate PGX Alpha-Zero (MuZero) trained agent playing strength against external
game database or application.  Includes interfaces for Awari and Connect4 lookup
databases, and external applications for Othello (Edax), Go-9x9x (KataGo, Pachi),
and Gardner Chess (Fairy-Stockfish).
"""

import os
import pickle
import random
from typing import TypeVar

import jax
import jax.numpy as jnp
import mctx
from flax import nnx
from omegaconf import OmegaConf
from opp_awari import AwariOraclePlayer
from opp_connect4 import Connect4OraclePlayer
from opp_gardner_chess import GardnerOraclePlayer
from opp_go_9x9 import GoOraclePlayer
from opp_othello import OthelloOraclePlayer
from pydantic import BaseModel

import pgx
from examples.alphazero.network import AZNet
from pgx.experimental import act_randomly

# Most agent/oracle configuration settings regarding playing strength
# and specific models used are currently configured via the environment.

ENV_ID = os.getenv("ENV_ID", "awari")
print("ENV_ID", ENV_ID)

NUM_CHANNELS = os.getenv("NUM_CHANNELS", "128")
num_channels = int(NUM_CHANNELS)
print("NUM_CHANNELS", num_channels)

NUM_LAYERS = os.getenv("NUM_LAYERS", "6")
num_layers = int(NUM_LAYERS)
print("NUM_LAYERS", num_layers)

NUM_ACTIONS = os.getenv("NUM_ACTIONS", "7")
num_actions = int(NUM_ACTIONS)
print("NUM_ACTIONS", num_actions)

# 000000.ckpt is almost random since it has not yet learned anything:
CHECKPOINT = os.getenv("CHECKPOINT", "./000000.ckpt")
print("CHECKPOINT", CHECKPOINT)

CHECKPOINT2 = os.getenv("CHECKPOINT2", "")
print("CHECKPOINT2", CHECKPOINT2)

NUM_SIMULATIONS = int(os.getenv("NUM_SIMULATIONS", "128"))
print("NUM_SIMULATIONS", NUM_SIMULATIONS)
NUM_SIMULATIONS = int(NUM_SIMULATIONS)

# Choose to use the oracle or a random eval for naive indicator
# of playing strenth
AWARI_ORACLE_ENABLE = True
CONNECT4_ORACLE_ENABLE = True
GARDNER_ORACLE_ENABLE = True
OTHELLO_ORACLE_ENABLE = True
GO_ORACLE_ENABLE = True

# Enable second oracle, i.e., checking relative strength of different oracle levels?
# AWARI_ORACLE_ENABLE2 = False
# AWARI_ORACLE_ENABLE2 = True
AWARI_ORACLE_ENABLE2 = bool(os.getenv("AWARI_ORACLE_ENABLE2", 0))
# print("AWARI_ORACLE_ENABLE2", AWARI_ORACLE_ENABLE2)
GARDNER_ORACLE_ENABLE2 = False
OTHELLO_ORACLE_ENABLE2 = False
GO_ORACLE_ENABLE2 = False
# GO_ORACLE_ENABLE2 = True
CONNECT4_ORACLE_ENABLE2 = bool(os.getenv("CONNECT4_ORACLE_ENABLE2", 0))
print("CONNECT4_ORACLE_ENABLE2", CONNECT4_ORACLE_ENABLE2)

# For oracles/agents that have a configurable mistake chance:
ORACLE_MISTAKE_CHANCE = os.getenv("ORACLE_MISTAKE_CHANCE", "0.20")
ORACLE_MISTAKE_CHANCE = float(ORACLE_MISTAKE_CHANCE)
print("ORACLE_MISTAKE_CHANCE", ORACLE_MISTAKE_CHANCE)
ORACLE_MISTAKE_MAX = os.getenv("ORACLE_MISTAKE_MAX", "3")
ORACLE_MISTAKE_MAX = int(ORACLE_MISTAKE_MAX)
print("ORACLE_MISTAKE_MAX", ORACLE_MISTAKE_MAX)

ORACLE_MISTAKE_CHANCE2 = os.getenv("ORACLE_MISTAKE_CHANCE2", "0.20")
ORACLE_MISTAKE_CHANCE2 = float(ORACLE_MISTAKE_CHANCE2)
print("ORACLE_MISTAKE_CHANCE2", ORACLE_MISTAKE_CHANCE2)
ORACLE_MISTAKE_MAX2 = os.getenv("ORACLE_MISTAKE_MAX2", "3")
ORACLE_MISTAKE_MAX2 = int(ORACLE_MISTAKE_MAX2)
print("ORACLE_MISTAKE_MAX2", ORACLE_MISTAKE_MAX2)

# For oracles/agents that have a configurable search depth or time:
ORACLE_DEPTH = os.getenv("ORACLE_DEPTH", "10")
ORACLE_DEPTH = int(ORACLE_DEPTH)
print("ORACLE_DEPTH", ORACLE_DEPTH)
ORACLE_TIME = os.getenv("ORACLE_TIME", "10")
ORACLE_TIME = int(ORACLE_TIME)
print("ORACLE_TIME", ORACLE_TIME)

ORACLE_DEPTH2 = os.getenv("ORACLE_DEPTH2", "10")
ORACLE_DEPTH2 = int(ORACLE_DEPTH2)
print("ORACLE_DEPTH2", ORACLE_DEPTH2)
ORACLE_TIME2 = os.getenv("ORACLE_TIME2", "10")
ORACLE_TIME2 = int(ORACLE_TIME2)
print("ORACLE_TIME2", ORACLE_TIME2)

RAND_SEED = int(os.getenv("RAND_SEED", "10"))
print("RAND_SEED", RAND_SEED)
RAND_SEED = int(RAND_SEED)

eval_batch_size = 1


class Config(BaseModel):
    # env_id: pgx.EnvId = "go_9x9"
    # env_id: pgx.EnvId = "awari"
    env_id: pgx.EnvId = ENV_ID
    # seed: int = 0
    seed: int = RAND_SEED
    # max_num_iters: int = 400
    max_num_iters: int = 2000
    # network params
    num_channels: int = num_channels
    num_layers: int = num_layers
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    # num_simulations: int = 32
    num_simulations: int = NUM_SIMULATIONS
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5

    class Config:
        extra = "forbid"


# config = Config()
conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)

env = pgx.make(config.env_id)

init = jax.jit(jax.vmap(env.init))  # vectorize and JIT-compile
step = jax.jit(jax.vmap(env.step))
act_randomly = jax.jit(act_randomly)

T = TypeVar("T")


def replicate(value: T, repeat: int) -> T:
    """Replicate along the first axis."""
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * repeat), value)


def _config_value(config, name):
    if isinstance(config, dict):
        return config[name]
    return getattr(config, name)


def make_az_model_replicated(filename):
    print("loading", filename)
    with open(filename, "rb") as f:
        d = pickle.load(f)

    model_config, model_data = d["config"], d["model"]
    model = AZNet(
        num_actions=env.num_actions,
        in_shape=env.observation_shape,
        num_channels=_config_value(model_config, "num_channels"),
        num_blocks=_config_value(model_config, "num_layers"),
        resnet_v2=_config_value(model_config, "resnet_v2"),
        rngs=nnx.Rngs(0),
    )

    if isinstance(model_data, tuple):
        # Checkpoint written by the legacy Haiku implementation.
        from pgx._src.baseline import _convert_az_params

        params, state = model_data
        _convert_az_params(
            model,
            params,
            state,
            resnet_v2=_config_value(model_config, "resnet_v2"),
        )
    else:
        nnx.update(model, model_data)

    model.eval()
    devices = jax.local_devices()
    graphdef, state = nnx.split(model)
    model = nnx.merge(graphdef, replicate(state, len(devices)))
    return model


def print_board(state, first_player, swap_color):  # noqa: C901
    if ENV_ID == "awari":
        board = state._x.board[0]
        print("   ", end="")
        for i in range(6):
            if first_player == 0:
                cell = board[11 - i]
            else:
                cell = board[5 - i]
            print(f"{cell:2d} ", end="")
        print()
        # board state orientation depends on who started first
        if first_player == 0:
            print(f"{board[23]:2d}", end="")
        else:
            print(f"{board[17]:2d}", end="")
        print("                   ", end="")
        if first_player == 0:
            print(f"{board[17]:2d}", end="")
        else:
            print(f"{board[23]:2d}", end="")
        print()
        print("   ", end="")
        for i in range(6):
            if first_player == 0:
                cell = board[i]
            else:
                cell = board[i + 6]
            print(f"{cell:2d} ", end="")
        print()
    elif ENV_ID == "othello":
        board = state._x.board[0]
        for y in range(8):
            for x in range(8):
                stone = board[8 * y + x]
                if stone == 0:
                    print(". ", end="")
                elif stone > 0:
                    if swap_color:
                        print("* ", end="")
                    else:
                        print("O ", end="")
                else:
                    if swap_color:
                        print("O ", end="")
                    else:
                        print("* ", end="")
            print("")
        print("")
    elif ENV_ID == "connect_four":
        board = state._x.board[0]
        for y in range(6):
            for x in range(7):
                stone = board[7 * y + x]
                if stone < 0:
                    print(". ", end="")
                elif stone > 0:
                    print("* ", end="")
                else:
                    print("O ", end="")
            print("")
        print("")
    elif ENV_ID == "go_9x9":
        board = state._x.board[0]
        for y in range(9):
            for x in range(9):
                stone = board[9 * y + x]
                if stone == 0:
                    print(". ", end="")
                elif stone > 0:
                    print("* ", end="")
                else:
                    print("O ", end="")
            print("")
        print("")
    elif ENV_ID == "gardner_chess":
        fen_str = state._to_fen()
        # print("fen:", fen_str)
        for row_fen in fen_str.split("/"):
            line = ""
            for char in row_fen:
                if char == " ":
                    # remaining non board data after this
                    break
                empty = 0
                if char.isdigit():
                    empty = int(char)
                elif char == "p":
                    piece = "♟"
                elif char == "P":
                    piece = "♙"
                elif char == "r":
                    piece = "♜"
                elif char == "R":
                    piece = "♖"
                elif char == "n":
                    piece = "♞"
                elif char == "N":
                    piece = "♘"
                elif char == "b":
                    piece = "♝"
                elif char == "B":
                    piece = "♗"
                elif char == "q":
                    piece = "♛"
                elif char == "Q":
                    piece = "♕"
                elif char == "k":
                    piece = "♚"
                elif char == "K":
                    piece = "♔"
                else:
                    piece = "?"
                if empty > 0:
                    line += ". " * empty
                else:
                    line += piece + " "
            print(line)
    elif ENV_ID == "tic_tac_toe":
        board = state._x.board[0]
        for y in range(3):
            for x in range(3):
                stone = board[3 * y + x]
                if stone < 0:
                    print(". ", end="")
                elif stone > 0:
                    print("X ", end="")
                else:
                    print("O ", end="")
            print("")
        print("")
    else:
        # just print full state datastructure
        print(state)


def print_board_csv(state, first_player, swap_color):
    if ENV_ID == "awari":
        board = state._x.board[0]
        # board state orientation depends on who started first
        for i in range(6):
            if first_player == 0:
                cell = board[11 - i]
            else:
                cell = board[5 - i]
            print(f"{cell},", end="")
        print(" ", end="")
        for i in range(6):
            if first_player == 0:
                cell = board[i]
            else:
                cell = board[i + 6]
            print(f"{cell},", end="")
        print(" ", end="")
        if first_player == 0:
            print(f"{board[23]},", end="")
        else:
            print(f"{board[17]},", end="")
        if first_player == 0:
            print(f"{board[17]},", end="")
        else:
            print(f"{board[23]},", end="")
        print(" ", end="")
    elif ENV_ID == "connect_four":
        board = state._x.board[0]
        if swap_color:
            this_player = 1
        else:
            this_player = 0
        for player in range(2):
            for y in range(6):
                for x in range(7):
                    stone = board[7 * y + x]
                    if stone < 0:
                        print("0,", end="")
                    elif stone > 0:
                        if player == this_player:
                            print("1,", end="")
                        else:
                            print("0,", end="")
                    else:
                        if player == this_player:
                            print("0,", end="")
                        else:
                            print("1,", end="")
            print(" ", end="")
        # print("")
    else:
        # just print full state datastructure
        print(state)


def print_game(env, g, model1, model2, result, first_reversed):
    verbose = True
    print("game", model1, "against", model2, "result", result)
    # For pgn conversion:
    if first_reversed:
        if result == 1:
            print("agent1_win_as_second")
        elif result == -1:
            print("agent2_win_as_first")
        else:
            print("draw result, agent2 started")
    else:
        if result == 1:
            print("agent1_win_as_first")
        elif result == -1:
            print("agent2_win_as_second")
        else:
            print("draw result, agent1 started")
    if True:
        # not default
        key = jax.random.PRNGKey(RAND_SEED)
        keys = jax.random.split(key, eval_batch_size)
        state = jax.vmap(env.init)(keys)
        first_player = state.current_player
        swap_color = state.current_player != first_player
        if verbose:
            print_board(state, first_player, swap_color)
        for i in range(len(g)):
            if g[i][1] != 0:
                print("total number of moves:", i)
                break
            move_id = g[i][0]
            action = jnp.int32([move_id])
            state = jax.vmap(env.step)(state, action)
            swap_color = state.current_player != first_player
            if verbose:
                print("move", i)
                print()
                print_board(state, first_player, swap_color)
                print()
    else:
        # Just print game length for sanity checks
        for i in range(len(g)):
            if g[i][1] != 0:
                print("total number of moves:", i)
                break


# PGX' internal simplified evaluation, based on examples/alphazero/train.py


def recurrent_fn(model: AZNet, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    # model: the NNX network (must be in eval mode)
    # state: embedding
    del rng_key

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    logits, value = model(state.observation)
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


@nnx.pmap(in_axes=0)
def evaluate_self(rng_key, my_model: AZNet, prev_model: AZNet):
    """A simplified evaluation by sampling. Only for debugging.
    Please use MCTS and run tournaments for serious evaluation."""
    my_player = 0

    key, subkey = jax.random.split(rng_key)
    # batch_size = config.selfplay_batch_size // num_devices
    batch_size = 1024
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        my_logits, _ = my_model(state.observation)
        opp_logits, _ = prev_model(state.observation)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size)))
    return R


# model evalution with mctx, based on examples/alphazero/train.py


def az_action_unbatched(state, model, rng_key: jnp.ndarray, num_simulations_per_move):
    graphdef, model_state = nnx.split(model)
    model_state = jax.tree.map(lambda x: x[0], model_state)
    model = nnx.merge(graphdef, model_state)
    model.eval()

    logits, value = model(state.observation)

    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

    def recurrent_fn_with_model(params, key, action, embedding):
        del params
        return recurrent_fn(model, key, action, embedding)

    key1, key2 = jax.random.split(rng_key)
    policy_output = mctx.gumbel_muzero_policy(
        # mctx threads `params` through JAX control flow, so it must be a pytree.
        params=(),
        rng_key=key1,
        root=root,
        recurrent_fn=recurrent_fn_with_model,
        # num_simulations=config.num_simulations,
        num_simulations=num_simulations_per_move,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )

    # return policy_output.action_weights,
    return policy_output.action


def agent_vs_oracle(  # noqa: C901
    # agent,
    # env: Enviroment,
    disable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
    num_games: int = 2,
    verbose: bool = False,
    # verbose: bool = True,
    oracle_mistake_chance: float = ORACLE_MISTAKE_CHANCE,
    oracle_mistake_max: int = ORACLE_MISTAKE_MAX,
    eval_moves: bool = True,
):
    """A number of games of agent vs oracle."""

    # debug is an extra verbose flag; use during development:
    debug = False

    key = jax.random.PRNGKey(RAND_SEED)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, eval_batch_size)

    oracle_mistake_chance2 = ORACLE_MISTAKE_CHANCE2
    oracle_mistake_max2 = ORACLE_MISTAKE_MAX2
    oracle_depth2 = ORACLE_DEPTH2
    oracle_time2 = ORACLE_TIME2

    model2 = None
    model1 = make_az_model_replicated(CHECKPOINT)

    if CHECKPOINT2 != "":
        model2 = make_az_model_replicated(CHECKPOINT2)

    result_agent_win = 0
    result_agent_win_as_first = 0
    result_agent_win_as_second = 0
    result_oracle_win = 0
    result_oracle_win_as_first = 0
    result_oracle_win_as_second = 0
    result_draw = 0

    # For othello try to add some randomization to initial phase to avoid playing
    # too many similar/same games which would perturb the statistics
    if ENV_ID == "othello":
        try:
            with open("openingslarge.txt", "r") as file:
                openings = file.readlines()
            if debug:
                print("openings[0]", openings[0])
                # e.g. f5d6c4d3c2b3b4b5
        except OSError:
            print("no Othello opening randomization support")
            openings = []

    external_player = None
    external_player2 = None
    if ENV_ID == "awari" and AWARI_ORACLE_ENABLE:
        external_player = AwariOraclePlayer(oracle_mistake_chance, oracle_mistake_max)
        if AWARI_ORACLE_ENABLE2:
            external_player2 = AwariOraclePlayer(oracle_mistake_chance2, oracle_mistake_max2)
    elif ENV_ID == "connect_four" and CONNECT4_ORACLE_ENABLE:
        external_player = Connect4OraclePlayer(oracle_mistake_chance, oracle_mistake_max)
        if CONNECT4_ORACLE_ENABLE2:
            external_player2 = Connect4OraclePlayer(oracle_mistake_chance2, oracle_mistake_max2)
    elif ENV_ID == "gardner_chess" and GARDNER_ORACLE_ENABLE:
        external_player = GardnerOraclePlayer(ORACLE_DEPTH, ORACLE_TIME)
        if GARDNER_ORACLE_ENABLE2:
            external_player2 = GardnerOraclePlayer(oracle_depth2, oracle_time2)
    elif ENV_ID == "othello" and OTHELLO_ORACLE_ENABLE:
        external_player = OthelloOraclePlayer(ORACLE_DEPTH, ORACLE_TIME)
        if OTHELLO_ORACLE_ENABLE2:
            external_player2 = OthelloOraclePlayer(oracle_depth2, oracle_time2)
    elif ENV_ID == "go_9x9" and GO_ORACLE_ENABLE:
        external_player = GoOraclePlayer(ORACLE_DEPTH, ORACLE_TIME)
        if GO_ORACLE_ENABLE2:
            external_player2 = GoOraclePlayer(oracle_depth2, oracle_time2)

    for game_id in range(num_games):
        # flip starting player
        agent_turn = (game_id + 1) % 2
        agent_started = (agent_turn % 2) == 0
        if verbose:
            print()
            print("game_id", game_id)
            print("agent_turn", agent_turn)
            print("agent_started", agent_started)

        eval_oracle = []
        eval_oracle2 = []
        eval_agent = []

        init_fn = jax.jit(jax.vmap(env.init))
        step_fn = jax.jit(jax.vmap(env.step))

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, eval_batch_size)
        state: pgx.State = init_fn(keys)

        # Some opponents don't support a direct board state input but can use
        # a move trace to reconstruct the proper position:
        move_trace = []

        first_player = state.current_player
        if verbose:
            print("first_player", first_player)

        # generate initial state based on one of the available traces giving
        # about equal value to both players, this avoids repeating the mostly
        # the same exchanges during evaluation
        if ENV_ID == "othello" and openings != []:
            initial_actions = openings[random.randint(0, len(openings) - 1)]
            if verbose:
                print("Initial move list:", initial_actions)
            for i in range(int(len(initial_actions) / 2)):
                action_letter = initial_actions[2 * i]
                action_number = initial_actions[2 * i + 1]
                move_id = (ord(action_number) - ord("1")) * 8 + (ord(action_letter) - ord("a"))
                if debug:
                    valid = state.legal_action_mask[0]
                    print("valid: ", end="")
                    for j in range(valid.size):
                        if valid[j]:
                            print(" " + str(j), end="")
                    print()
                    print("move", i, ":", action_letter + action_number, " = ", move_id)
                action = jnp.int32([move_id])
                state = step_fn(state, action)
                if debug:
                    print(
                        "state:", state, "observation", state.observation, "legal_action_mask", state.legal_action_mask
                    )
                swap_color = state.current_player != first_player
                if verbose:
                    print_board(state, first_player, swap_color)
                move_trace.append(move_id)

        # limit number of moves to 256 to protect against infinite loops,
        # which is also the default PGX setting
        for move in range(256):
            if verbose:
                print()
                print(f"Move {move}")
                print("======")
                print()

            if state.terminated.all():
                break

            dont_use_oracle = (
                (((move % 2) == agent_turn) or external_player is None) and (external_player2 is None)
            ) or (model2 is not None)
            if dont_use_oracle:
                if debug:
                    print("current_player", state.current_player)
                    print("legal_action_mask", state.legal_action_mask)
                    print("rewards", state.rewards)

                if ((external_player is None) or (model2 is None)) and (move % 2 != agent_turn):
                    action = act_randomly(subkey, state.legal_action_mask)
                    if verbose:
                        print("RANDOM action:", action)
                else:
                    # do alpha zero action, based on model whose turn it is and mcts search
                    key, subkey = jax.random.split(key)
                    if model2 is not None and move % 2 != agent_turn:
                        action = az_action_unbatched(state, model2, subkey, num_simulations_per_move)
                    else:
                        action = az_action_unbatched(state, model1, subkey, num_simulations_per_move)

                    if debug:
                        print("az_action_unbatched", action, "num_simulations_per_move", num_simulations_per_move)

                    if ENV_ID == "awari" and eval_moves:
                        action_id = int(action[0])
                        eval_agent.append(external_player.evaluate(state, action_id, debug))

                if verbose:
                    print("model action", action[0])
            else:
                # use external player for move
                swap_color = state.current_player != first_player
                if debug:
                    external_player.print_oracle_status(state, move_trace, swap_color)
                    external_player.print_valid(state, swap_color, move_trace, verbose)
                if ((move % 2) == agent_turn) and (external_player2 is not None):
                    # agent_turn is also used for oracle2
                    action, scores = external_player2.play(state, swap_color, move_trace, debug)
                else:
                    # action = external_player.play(state, swap_color, move_trace, debug)
                    action, scores = external_player.play(state, swap_color, move_trace, verbose)
                if debug:
                    print(f"# External opponent selected action {action}")
                if action < 0 or action > len(state.legal_action_mask[0]) or not state.legal_action_mask[0][action]:
                    # warn external agent returned invalid action, and quit game
                    print("INVALID action", action, "mask", state.legal_action_mask[0])
                    break
                action = jnp.int32([action])
                if verbose:
                    print("external action", action[0])
                if ENV_ID == "awari" and eval_moves:
                    action_id = int(action[0])
                    eval_oracle.append(external_player.evaluate(state, action_id, debug))
                if ENV_ID == "connect_four" or ENV_ID == "awari":
                    print("MOVE", move, end="")
                    print(", ", end="")
                    print_board_csv(state, first_player, swap_color)
                    for i in range(len(scores)):
                        print(scores[i], end="")
                        if i < len(scores) - 1:
                            print(",", end="")
                    print("")

            state = step_fn(state, action)
            move_trace.append(int(action[0]))
            if debug:
                print("trace", move_trace)
            if verbose:
                swap_color = state.current_player != first_player
                print_board(state, first_player, swap_color)
        else:
            print("Exceeded max move count")

        if verbose:
            print("current_player", state.current_player)
            # NOTE: the "color" and "current_player" may be reversed, which impacts rewards
            print("first_player", first_player)

        # rewards are indexed by player id, but player ids are independent
        # of who started, so this needs to be accounted for since we want
        # to report rewards relative to who played first
        rewards = state.rewards
        should_flip = first_player != 0
        if should_flip:
            if verbose:
                print("flipping rewards", rewards)
            rewards = jnp.flip(rewards)
            if verbose:
                print("rewards now", rewards)

        if verbose:
            print()
            print("Final board")
            print("===========")
            print()
            # env.render()
            # print("state", state)
            swap_color = state.current_player != first_player
            print_board(state, first_player, swap_color)
            print()
            print("rewards[0,0]", rewards[0, 0])

        if rewards[0, 0] == 1:
            if agent_started:
                result_agent_win += 1
                result_agent_win_as_first += 1
                print("agent_win_as_first")
            else:
                result_oracle_win += 1
                result_oracle_win_as_first += 1
                print("oracle_win_as_first")
        elif rewards[0, 0] == -1:
            if agent_started:
                result_oracle_win += 1
                result_oracle_win_as_second += 1
                print("oracle_win_as_second")
            else:
                result_agent_win += 1
                result_agent_win_as_second += 1
                print("agent_win_as_second")
        else:
            result_draw += 1
            print("draw result")

        if ENV_ID == "awari" and eval_moves:
            if eval_agent != []:
                print("eval agent", eval_agent)
            print("eval oracle", eval_oracle)
            if eval_oracle2 != []:
                print("eval oracle2", eval_oracle2)

    print("agent_win", result_agent_win, "oracle_win", result_oracle_win, "draw", result_draw)
    print(
        "agent_win_as_first",
        result_agent_win_as_first,
        "agent_win_as_second",
        result_agent_win_as_second,
        "oracle_win_as_first",
        result_oracle_win_as_first,
        "oracle_win_as_second",
        result_oracle_win_as_second,
        "draw",
        result_draw,
    )


mode = "agent_vs_oracle"

if __name__ == "__main__":
    verbose = True
    if mode == "agent_vs_oracle":
        agent_vs_oracle(
            num_simulations_per_move=NUM_SIMULATIONS,
            # num_games=2,
            num_games=20,
            # num_games=200,
            verbose=verbose,
        )
    else:
        # compare two given models directly, without MCTS
        key = jax.random.PRNGKey(RAND_SEED)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, 1)  # num_devices

        model1 = make_az_model_replicated(CHECKPOINT)
        model2 = make_az_model_replicated(CHECKPOINT2)

        R = evaluate_self(keys, model1, model2)
        print("number total", R.size)
        opp = "agent2"
        print(f"eval/{opp}/avg_R", R.mean().item())
        print(f"eval/{opp}/win_rate", ((R > 0).sum() / R.size).item())
        print(f"eval/{opp}/draw_rate", ((R == 0).sum() / R.size).item())
        print(f"eval/{opp}/lose_rate", ((R < 0).sum() / R.size).item())
