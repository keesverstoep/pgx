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

import datetime
import os
import pickle
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx
import optax
from flax import nnx
from network import AZNet
from omegaconf import OmegaConf
from pydantic import BaseModel

import pgx
from pgx.experimental import auto_reset

devices = jax.local_devices()
num_devices = len(devices)

RAND_SEED = int(os.getenv("RAND_SEED", "10"))

NUM_LAYERS = os.getenv("NUM_LAYERS", "6")
# NUM_LAYERS = os.getenv("NUM_LAYERS", "12")
num_layers = int(NUM_LAYERS)
print("NUM_LAYERS", num_layers)

NUM_CHANNELS = os.getenv("NUM_CHANNELS", "128")
num_channels = int(NUM_CHANNELS)
print("NUM_CHANNELS", num_channels)

LEARNING_RATE = os.getenv("LEARNING_RATE", "0.001")
print("LEARNING_RATE", LEARNING_RATE)
learning_rate = float(LEARNING_RATE)


class Config(BaseModel):
    # env_id: pgx.EnvId = "go_9x9"
    # HACK: to get the experimental eval working:
    env_id: pgx.EnvId = "gardner_chess"
    # seed: int = 0
    seed: int = RAND_SEED
    # max_num_iters: int = 400
    # changed: look at training tail
    max_num_iters: int = 10000
    # network params
    # default:
    # For the ResNet experiments:
    # num_channels: int = 128
    num_channels: int = num_channels
    # num_layers: int = 6
    # num_layers: int = 12
    num_layers: int = num_layers
    # For the MLP experiments:
    # num_channels: int = 256
    # num_layers: int = 3
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    # gumbel version does not need many simulations:
    num_simulations: int = 32
    # num_simulations: int = 300
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    # learning_rate: float = 0.001
    # reduced, fine tuning:
    # learning_rate: float = 0.0001
    learning_rate: float = learning_rate
    # eval params
    eval_interval: int = 5

    # added:
    window_size: int = 1_000_000
    shuffle_window: bool = True
    eval_batch_size: int = 32
    n_training_pass: int = 1
    param_count: int = 0  # will be computed
    baseline: str = "none"

    class Config:
        extra = "forbid"


conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)

env = pgx.make(config.env_id)

baseline = pgx.make_baseline_model(config.env_id + "_v0")


def make_model(seed: int = 0) -> AZNet:
    return AZNet(
        num_actions=env.num_actions,
        in_shape=env.observation_shape,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
        rngs=nnx.Rngs(seed),
    )


def make_optimizer(model: nnx.Module) -> nnx.Optimizer:
    return nnx.Optimizer(model, optax.adam(learning_rate=config.learning_rate), wrt=nnx.Param)


def replicate(tree):
    """Replicate every array in ``tree`` along a leading device axis."""
    return jax.tree.map(lambda x: jnp.stack([x] * num_devices), tree)


def replicate_module(module):
    graphdef, state = nnx.split(module)
    return nnx.merge(graphdef, replicate(state))


def unreplicate(tree):
    """Take the first replica of every array in ``tree``."""
    return jax.tree.map(lambda x: x[0], tree)


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


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@nnx.pmap(in_axes=0)
def selfplay(model: AZNet, rng_key: jnp.ndarray) -> SelfplayOutput:
    batch_size = config.selfplay_batch_size // num_devices

    def recurrent_fn_with_model(params, key, action, state):
        del params
        return recurrent_fn(model, key, action, state)

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        logits, value = model(state.observation)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            # mctx threads `params` through JAX control flow, so it must be a
            # pytree. The network is closed over by `recurrent_fn_with_model`.
            params=(),
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn_with_model,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def loss_fn(model: AZNet, samples: Sample):
    logits, value = model(samples.obs)

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)  # mask if the episode is truncated

    return policy_loss + value_loss, (policy_loss, value_loss)


@nnx.pmap(axis_name="i", in_axes=0)
def train(model: AZNet, optimizer: nnx.Optimizer, data: Sample):
    (loss, (policy_loss, value_loss)), grads = nnx.value_and_grad(
        loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True
    )(model, data)
    del loss
    grads = jax.lax.pmean(grads, axis_name="i")
    optimizer.update(model, grads)
    return policy_loss, value_loss


@nnx.pmap(in_axes=0)
def evaluate(rng_key, my_model: AZNet):
    """A simplified evaluation by sampling. Only for debugging.
    Please use MCTS and run tournaments for serious evaluation."""
    my_player = 0

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        my_logits, _ = my_model(state.observation)
        opp_logits, _ = baseline(state.observation)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size)))
    return R


@nnx.pmap(in_axes=0)
def evaluate_self(rng_key, my_model: AZNet, prev_model: AZNet):
    """A simplified evaluation by sampling. Only for debugging.
    Please use MCTS and run tournaments for serious evaluation."""
    my_player = 0

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
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


@nnx.pmap(in_axes=0)
def evaluate_random(rng_key, my_model: AZNet):
    """A simplified evaluation against a random agent as a sanity check."""
    my_player = 0

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        my_logits, _ = my_model(state.observation)
        # random selection between all legal actions
        opp_logits = jnp.ones_like(state.legal_action_mask)
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)

        # mask illegal actions (borrowed from alphax)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        # next changes borrowed from alphax
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        state = jax.vmap(env.step)(state, action, keys)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size)))
    return R


def _config_value(config, name):
    if isinstance(config, dict):
        return config[name]
    return getattr(config, name)


# to pickup from a previous checkpoint, from eval/eval-agents.py
def make_az_model_replicated(filename):
    print("loading", filename)
    with open(filename, "rb") as f:
        d = pickle.load(f)

    model_config = d["config"]
    model = AZNet(
        num_actions=env.num_actions,
        in_shape=env.observation_shape,
        num_channels=_config_value(model_config, "num_channels"),
        num_blocks=_config_value(model_config, "num_layers"),
        resnet_v2=_config_value(model_config, "resnet_v2"),
        rngs=nnx.Rngs(0),
    )

    model_data = d["model"]
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
        optimizer = make_optimizer(model)
    else:
        nnx.update(model, model_data)
        optimizer = make_optimizer(model)
        opt_state = d.get("opt_state")
        if opt_state is not None:
            nnx.update(optimizer, opt_state)

    model.eval()
    iteration = d.get("iteration", 0)
    model = replicate_module(model)
    optimizer = replicate_module(optimizer)
    return model, optimizer, iteration


def save_checkpoint(path, model: AZNet, optimizer: nnx.Optimizer, iteration, frames, hours, rng_key):
    model_state = unreplicate(jax.device_get(nnx.state(model)))
    opt_state = unreplicate(jax.device_get(nnx.state(optimizer)))
    with open(path, "wb") as f:
        pickle.dump(
            {
                "config": config.model_dump(),
                "rng_key": rng_key,
                "model": model_state,
                "opt_state": opt_state,
                "iteration": iteration,
                "frames": frames,
                "hours": hours,
                "pgx.__version__": pgx.__version__,
                "env_id": env.id,
                "env_version": env.version,
            },
            f,
        )


if __name__ == "__main__":
    # wandb.init(project="pgx-az", config=config.model_dump())

    # to resolve jax FutureWarning issues:
    import warnings

    warnings.filterwarnings("error")

    iteration: int = 0

    # Initialize model and optimizer, possibly from previous checkpoint
    CHECKPOINT = os.getenv("CHECKPOINT", "")
    if CHECKPOINT != "":
        print("init from checkpoint", CHECKPOINT)
        model, optimizer, iteration = make_az_model_replicated(CHECKPOINT)
        # needed for self eval:
        model_prev = nnx.clone(model)
        model_best = nnx.clone(model)
    else:
        model = make_model(config.seed)
        optimizer = make_optimizer(model)
        # replicates to all devices
        model = replicate_module(model)
        optimizer = replicate_module(optimizer)
        # added for self-eval:
        model_prev = None
        model_best = None

    # Prepare checkpoint dir
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"{config.env_id}_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize logging dict
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    rng_key = jax.random.PRNGKey(config.seed)
    while True:
        model.eval()
        if iteration % config.eval_interval == 0 and baseline is not None:
            # Evaluation
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            R = evaluate(keys, model)
            log.update(
                {
                    "eval/vs_baseline/avg_R": R.mean().item(),
                    "eval/vs_baseline/win_rate": ((R == 1).sum() / R.size).item(),
                    "eval/vs_baseline/draw_rate": ((R == 0).sum() / R.size).item(),
                    "eval/vs_baseline/lose_rate": ((R == -1).sum() / R.size).item(),
                }
            )
        # if iteration % config.eval_interval == 0 and baseline == None and model_prev != None:
        if iteration % config.eval_interval == 0 and model_prev is not None:
            # Self-evaluation against previous iteration
            model_prev.eval()
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            R = evaluate_self(keys, model, model_prev)
            log.update(
                {
                    "eval/vs_prev/avg_R": R.mean().item(),
                    "eval/vs_prev/win_rate": ((R == 1).sum() / R.size).item(),
                    "eval/vs_prev/draw_rate": ((R == 0).sum() / R.size).item(),
                    "eval/vs_prev/lose_rate": ((R == -1).sum() / R.size).item(),
                }
            )

            # Self-evaluation against best iteration up to now
            model_best.eval()
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            R = evaluate_self(keys, model, model_best)
            log.update(
                {
                    "eval/vs_best/avg_R": R.mean().item(),
                    "eval/vs_best/win_rate": ((R == 1).sum() / R.size).item(),
                    "eval/vs_best/draw_rate": ((R == 0).sum() / R.size).item(),
                    "eval/vs_best/lose_rate": ((R == -1).sum() / R.size).item(),
                    "eval/vs_best/new_best": 1 if (R.mean().item() > 0.0) else 0,
                }
            )
            if R.mean().item() > 0.0:
                model_best = nnx.clone(model)

            # Also evaluation against random agent as a sanity check
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            R = evaluate_random(keys, model)
            log.update(
                {
                    "eval/vs_random/avg_R": R.mean().item(),
                    "eval/vs_random/win_rate": ((R == 1).sum() / R.size).item(),
                    "eval/vs_random/draw_rate": ((R == 0).sum() / R.size).item(),
                    "eval/vs_random/lose_rate": ((R == -1).sum() / R.size).item(),
                }
            )

        if iteration % config.eval_interval == 0:
            # Store checkpoints
            save_checkpoint(
                os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"),
                model,
                optimizer,
                iteration,
                frames,
                hours,
                rng_key,
            )
            model_prev = nnx.clone(model)
            if model_best is None:
                model_best = nnx.clone(model)

        print(log)
        # wandb.log(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Selfplay
        model.eval()
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(model, keys)
        samples: Sample = compute_loss_input(data)

        # Shuffle samples and make minibatches
        samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree.map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree.map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree.map(lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples)

        # Training
        model.train()
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree.map(lambda x: x[i], minibatches)
            policy_loss, value_loss = train(model, optimizer, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        )
