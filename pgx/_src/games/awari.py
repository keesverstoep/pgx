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
#

#
# Author: Kees Verstoep, Vrije Universiteit Amsterdam, 2024
#
# - This is a PGX port of a previous JAX-based implementation of Awari.
# - See https://en.wikipedia.org/wiki/Oware for background on the game
# - The full game state has been computed previously and is available here:
#      https://research.vu.nl/en/datasets/awari-game-score-database
#   - Interfacing with this database is currently being investigated
#     to compare training progress against ground truth for the game.
#

import jax
import jax.numpy as jnp

from typing import NamedTuple, Optional

from jax import lax
import numpy as np

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

from typing import Tuple

import chex

# Array used for implementing the sowing logic, which has
# to skip the pit from which the stones are sown from.
next = jnp.int32([
    # Added extra column 12 so a pass move can "pass through"
    # the regular sowing code without modifying the board:
    # 0  1  2  3  4  5  6  7   8   9  10  11 12
    [ 1, 2, 3, 4, 5, 7, 7, 8,  9, 10, 11, 0, 12, ],
    [ 1, 2, 3, 4, 5, 6, 8, 8,  9, 10, 11, 0, 12, ],
    [ 1, 2, 3, 4, 5, 6, 7, 9,  9, 10, 11, 0, 12, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 0, 12, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8,  9, 11, 11, 0, 12, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8,  9, 10,  0, 0, 12, ],
])

# Currently uses the following simple board representation:
# - pit[0:6]  : regular pits by player0 (canonical: the one to move)
# - pit[6:12] : regular pits by player1 (canonical: the one not to move)
# - pit[12] :   an empty pit for the sowing logic in case of passing move
# - pit[17]:    home pit of player0
# - pit[23]:    home pit of player1
# Other pit positions are unused.

PIT_MAX = 24
PIT_HOME_0 = 17
PIT_HOME_1 = 23

NUM_STONES = 48

def select_tree(pred: jnp.ndarray, a, b):
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
    return jax.tree_util.tree_map(jax.tree_util.Partial(jax.lax.select, pred), a, b)

INIT_BOARD = jnp.int32([
    4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
])

NUM_ROWS = 1
NUM_COLS = 24

class GameState(NamedTuple):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    # 6 regular moves and (no) pass:
    legal_action_mask: Array = jnp.array([ True, True, True, True, True, True, False ])
    board: Array = INIT_BOARD
    # winner is -1 (no winner), or player id of winner (0 or 1)
    winner: Array = jnp.int32(-1)
    # compare current_player to first_player to see if board needs flip
    first_player: Array = jnp.int32(0)

    def init(self, current_player):
        self.current_player = current_player
        self.first_player = current_player

class Game:
    def init(self, current_player) -> GameState:
        return GameState(current_player)

    def step(self, state: GameState, action: Array) -> GameState:
        return _step(state, action)

    def observe(self, state: GameState, player_id: Array) -> Array:
        return _observe(state, player_id)

    def is_terminal(self, state: GameState) -> Array:
        remaining_stones = lax.slice_in_dim(state.board, 0, 12).sum()
        return (state.winner >= 0) | (state.terminated) | (remaining_stones == 0)


def _invalid_actions(state_board, flipped_state_board, needs_flip) -> Array:
    # for awari can only select pit that has stones
    # - may only select pit to sow that eradicates opponent
    #   when that is the only move left
    # For this we will have have to try all moves:
    # - if some them leaves the opponent with a move and others
    #   do not, only the ones that do are valid

    pieces = jnp.where(needs_flip, flipped_state_board, state_board)
    child = jnp.where(needs_flip, state_board, flipped_state_board)

    remaining_other = jnp.int32([0, 0, 0, 0, 0, 0])

    def cond1(state):
        # i, _, _ = state
        i, _ = state
        return (i < 6)

    def body1(state):
        i, rem_other = state
        newboard = sow_canonical(pieces, child, i)
        # returns *mirrored* canonical board, so look at 0..5 for remaining opp stones
        # Only consider remaining opp stones if this is actually a possible move
        rem_other = jnp.where(pieces[i] == 0, rem_other,
                              rem_other.at[i].set(lax.slice_in_dim(newboard, 0, 6).sum()))
        return (i + 1, rem_other)

    state = tuple([0, remaining_other])
    lasti, remaining_other = lax.while_loop(cond_fun=cond1, body_fun=body1, init_val=state)

    remainingall_other = lax.slice_in_dim(remaining_other, 0, 6).sum()

    # regular invalidations, reserve space for pass:
    invalidthis = (pieces[0:7] == 0)

    # if thisplayer and remainingall other is zero, then any nonempty pit is fine
    # if thisplayer and some other remaining pits are zero, these moves are invalid

    def cond2(state):
        i, _ = state
        return (i < 6)
    
    def body2(state):
        i, invalidthis_loc = state
        this_move_invalid_loc = jnp.logical_and(remainingall_other > 0, remaining_other[i] == 0)
        invalidthis_loc = invalidthis_loc.at[i].set(jnp.logical_or(invalidthis_loc[i], this_move_invalid_loc))
        return (i + 1, invalidthis_loc)

    state2 = tuple([0, invalidthis])
    lasti, invalidthis = lax.while_loop(cond_fun=cond2, body_fun=body2, init_val=state2)

    # allow pass when no moves
    invalid = invalidthis
    invalid1 = invalid.at[6].set(True)
    invalid2 = invalid1.at[6].set(jnp.logical_not(jnp.all(invalid1)))

    return invalid2

def sow_canonical(pieces: chex.Array, child: chex.Array, move: int):
    pit1 = move
    seeds = pieces[pit1]
    pit2 = pit1 + 6
    capture = 0
    oldseeds = seeds
    next_ptr = next[pit1]

    # take away seeds
    pieces = child.at[pit2].set(0)

    # sow them
    def cond1(state):
        seeds_cur, pit2_cur, pieces_cur = state
        return (seeds_cur > 0)

    def body1(state):
        seeds_cur, pit2_cur, pieces_cur = state
        pit2_cur = next_ptr[pit2_cur]
        temp = jnp.add(pieces_cur[pit2_cur], 1)
        pieces2 = pieces_cur.at[pit2_cur].set(temp)
        return (seeds_cur - 1, pit2_cur, pieces2)

    state = tuple([seeds, pit2, pieces])
    seeds, pit2, pieces2 = lax.while_loop(cond_fun=cond1, body_fun=body1, init_val=state)
    pieces = pieces2

    # deal with captures
    def cond2(state):
        pit2_cur, captures_cur, pieces_cur = state
        ok1 = (pit2_cur < 6)
        ok2 = (pit2_cur >= 0)
        ok3 = (pieces_cur[pit2_cur] == 2)
        ok4 = (pieces_cur[pit2_cur] == 3)
        return jnp.logical_and(ok1, jnp.logical_and(ok2, jnp.logical_or(ok3, ok4)))

    def body2(state):
        pit2_cur, captures_cur, pieces_cur = state
        captures_cur = jnp.add(captures_cur, pieces_cur[pit2_cur])
        pieces_new = pieces_cur.at[pit2_cur].set(0)
        return (pit2_cur - 1, captures_cur, pieces_new)

    captures = 0
    state = tuple([pit2, captures, pieces])
    pit2_new, captures_new, pieces_new = lax.while_loop(cond_fun=cond2, body_fun=body2, init_val=state)
    captures = captures_new

    # canonical board, so captures are for current player orientation,
    # but the resulting board is mirrored, so we need to use the other home pit here
    ret_pieces = pieces_new.at[PIT_HOME_1].set(jnp.add(pieces_new[PIT_HOME_1], captures))
    return ret_pieces

def sow(state, pieces: chex.Array, child: chex.Array, move: int):
    # pieces_new = self.sow_canonical(pieces, child, move)
    pieces_new = sow_canonical(pieces, child, move)

    # now return the board in the standard first_player orientation
    # Note that sowing left them in a mirrored bord
    x0 = lax.slice_in_dim(pieces_new, 0, 6)
    x1 = lax.slice_in_dim(pieces_new, 6, 12)
    x2 = lax.slice_in_dim(pieces_new, 12, 18)
    x3 = lax.slice_in_dim(pieces_new, 18, 24)
    pieces_mirror = jax.lax.concatenate([x1, x0, x3, x2], 0)
    thisplayer = (state.current_player == state.first_player)
    ret_pieces = jnp.where(thisplayer, pieces_mirror, pieces_new)
    return ret_pieces

def _step(state, action: chex.Array) -> Tuple["AwariGame", chex.Array]:
    # need checks for passing, it's only allowed when no other options exist
    pass_move = (action == 6)

    _board = state.board
    x0 = lax.slice_in_dim(_board, 0, 6)
    x1 = lax.slice_in_dim(_board, 6, 12)
    x2 = lax.slice_in_dim(_board, 12, 18)
    x3 = lax.slice_in_dim(_board, 18, 24)
    _mirror = jax.lax.concatenate([x1, x0, x3, x2], 0)

    needs_flip = (state.current_player != state.first_player)
    # thisplayer/otherplayer for compatibility with the a0-jax approach
    thisplayer = (state.current_player == state.first_player)
    otherplayer = (state.current_player != state.first_player)
    pieces = jnp.where(thisplayer, _board, _mirror)
    child = jnp.where(thisplayer, _mirror, _board)

    # Check for invalid moves: only allowed to select pit that has seeds
    invalid_move = jnp.logical_and(action < 6, pieces[action] == 0)

    # Check if a passing is valid:
    invalid_pass0 = jnp.logical_and(thisplayer, jnp.logical_and(pass_move, jnp.any(x0)))
    invalid_pass1 = jnp.logical_and(otherplayer, jnp.logical_and(pass_move, jnp.any(x1)))
    invalid_pass = jnp.logical_or(invalid_pass0, invalid_pass1)
    invalid_move = jnp.logical_or(invalid_move, invalid_pass)

    # - check if move that leaves the opponent no stones is the only option,
    #   otherwise it is invalid

    # trick to skip the sowing code when doing a pass move: pit 12 is always empty
    move = jnp.where(pass_move, 12, action)
    
    # sow canonical as we are still need some operations, and it is
    # easier to do this on a board relative to the current player
    newboard_flipped = sow_canonical(pieces, child, move)
    x0 = lax.slice_in_dim(newboard_flipped, 0, 6)
    x1 = lax.slice_in_dim(newboard_flipped, 6, 12)
    x2 = lax.slice_in_dim(newboard_flipped, 12, 18)
    x3 = lax.slice_in_dim(newboard_flipped, 18, 24)
    newboard = jax.lax.concatenate([x1, x0, x3, x2], 0)

    # after (forced) pass move, all remaining seeds go to the opponent
    pass_move = jnp.where(invalid_move, False, pass_move)
    remaining_pieces = lax.slice_in_dim(newboard, 0, 12)
    remaining_pieces_count = remaining_pieces.sum()
    remaining_pits = jnp.int32([0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0])
    remaining_pits = remaining_pits.at[PIT_HOME_0].set(newboard[PIT_HOME_0])
    remaining_pits = remaining_pits.at[PIT_HOME_1].set(newboard[PIT_HOME_1])

    # newboard is oriented towards current player, so use that below
    new_state_board = select_tree(state.terminated, _board, newboard)

    new_state_board = select_tree(pass_move,
            remaining_pits.at[PIT_HOME_1].add(remaining_pieces_count),
            new_state_board)

    # set winner if one player has a majority of stones
    winner0 = (new_state_board[PIT_HOME_0] > NUM_STONES / 2)
    winner1 = (new_state_board[PIT_HOME_1] > NUM_STONES / 2)

    # note that current board on which winning status is based is canonical
    new_state_winner = jnp.where(winner0, state.current_player, -1)
    new_state_winner = jnp.where(winner1, 1 - state.current_player, new_state_winner)

    # exceeding step counts is done at higher layer
    # new_state_terminated = jnp.logical_or(state.terminated, state._step_count >= 126)
    new_state_terminated = state.terminated

    # termination due to win/loss
    new_state_terminated = jnp.logical_or(new_state_terminated, new_state_winner != -1)

    # termination after pass due to no move left
    new_state_terminated = jnp.logical_or(new_state_terminated, pass_move)

    # termination due to invalid move, which also causes negative reward
    new_state_terminated = jnp.logical_or(new_state_terminated, invalid_move)

    # set rewards
    new_state_reward = jax.lax.cond(new_state_winner == -1,
                           lambda: jnp.zeros(2, jnp.float32),
                           lambda: jnp.float32([-1, -1]).at[new_state_winner].set(1))

    x0 = lax.slice_in_dim(new_state_board, 0, 6)
    x1 = lax.slice_in_dim(new_state_board, 6, 12)
    x2 = lax.slice_in_dim(new_state_board, 12, 18)
    x3 = lax.slice_in_dim(new_state_board, 18, 24)
    flipped_new_state_board = jax.lax.concatenate([x1, x0, x3, x2], 0)

    returned_new_state_board = jnp.where(needs_flip, flipped_new_state_board, new_state_board)
    returned_new_state_board_mirror = jnp.where(needs_flip, new_state_board, flipped_new_state_board)

    # Compute actions are for the next state board, i.e. flipped so seen from the other player
    illegal_actions = _invalid_actions(flipped_new_state_board, new_state_board, False)
    legal_actions = jnp.logical_not(illegal_actions)

    return state._replace(
         current_player=1 - state.current_player,
         legal_action_mask=legal_actions,
         rewards=new_state_reward,
         terminated=new_state_terminated,
         board=returned_new_state_board,
         winner=new_state_winner,
    )

def _observe(state, player_id) -> chex.Array:
    _board = state.board
    x0 = lax.slice_in_dim(_board, 0, 6)
    x1 = lax.slice_in_dim(_board, 6, 12)
    x2 = lax.slice_in_dim(_board, 12, 18)
    x3 = lax.slice_in_dim(_board, 18, 24)
    _mirror = jax.lax.concatenate([x1, x0, x3, x2], 0)
    ret_board = jnp.where(player_id != state.first_player, _mirror, _board)

    # put own and other home pit state in full separate layers
    x0 = lax.slice_in_dim(ret_board, 0, 6)
    x1 = lax.slice_in_dim(ret_board, 6, 12)
    own = ret_board[17]
    other = ret_board[23]
    x2 = jnp.int32([own, own, own, own, own, own])
    x3 = jnp.int32([other, other, other, other, other, other])
    zeros = jnp.zeros(6, dtype=jnp.int32)

    layer0 = jax.lax.concatenate([x0, zeros], 0).reshape(2, 6)
    layer1 = jax.lax.concatenate([zeros, x1], 0).reshape(2, 6)
    layer2 = jax.lax.concatenate([x2, x2], 0).reshape(2, 6)
    layer3 = jax.lax.concatenate([x3, x3], 0).reshape(2, 6)
    return jnp.stack([layer0, layer1, layer2, layer3], -1)

def render(state) -> None:
    """Render the game on screen."""
    # TODO: pretty print
    print("awari state:", state)

