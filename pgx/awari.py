# Copyright 2023 The Pgx Authors. All Rights Reserved.
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

# This top level PGX interaction code is mostly based on other games
# included with PGX like connect4 that have a split implementation.

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.games.awari import Game, GameState
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

# Set max steps just below 128 since this is the default
# for some training frameworks using PGX
MAX_TERMINATION_STEPS = 126

@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((2, 6, 4), dtype=jnp.int32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.array([ True, True, True, True, True, True, False ]) 
    _step_count: Array = jnp.int32(0)
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "awari"


class Awari(core.Env):
    def __init__(self):
        super().__init__()
        self._game = Game()

    def _init(self, key: PRNGKey) -> State:
        current_player = jnp.int32(jax.random.bernoulli(key))
        return State(current_player=current_player, _x=self._game.init(current_player))  # type:ignore

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        x = self._game.step(state._x, action)
        state = state.replace(  # type: ignore
            current_player=1 - state.current_player,
            _x=x,
        )
        assert isinstance(state, State)
        legal_action_mask = state._x.legal_action_mask
        terminated = (state._step_count >= MAX_TERMINATION_STEPS)
        terminated = jnp.logical_or(terminated, self._game.is_terminal(state._x))
        rewards = state._x.rewards
        rewards = jax.lax.select(terminated, rewards, jnp.zeros(2, jnp.float32))
        return state.replace(  # type: ignore
            legal_action_mask=legal_action_mask,
            rewards=rewards,
            terminated=terminated,
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return self._game.observe(state._x, player_id)

    @property
    def id(self) -> core.EnvId:
        return "awari"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2
