"""This PPO implementation is modified from PureJaxRL:

  https://github.com/luchris429/purejaxrl

Please refer to their work if you use this example in your research."""

import pickle
import sys
import time
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax
import pgx
import wandb
from flax import nnx
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel


class PPOConfig(BaseModel):
    env_name: Literal[
        "minatar-breakout",
        "minatar-freeway",
        "minatar-space_invaders",
        "minatar-asterix",
        "minatar-seaquest",
    ] = "minatar-breakout"
    seed: int = 0
    lr: float = 0.0003
    num_envs: int = 4096
    num_eval_envs: int = 100
    num_steps: int = 128
    total_timesteps: int = 20000000
    update_epochs: int = 3
    minibatch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    wandb_project: str = "pgx-minatar-ppo"
    save_model: bool = False

    class Config:
        extra = "forbid"


args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args, file=sys.stderr)
env = pgx.make(str(args.env_name))


num_updates = args.total_timesteps // args.num_envs // args.num_steps
num_minibatches = args.num_envs * args.num_steps // args.minibatch_size


def _avg_pool2x2(x):
    """Match ``hk.avg_pool(x, (2, 2), (2, 2), "VALID")`` for NHWC inputs."""
    batch, height, width, channels = x.shape
    width2, channels2 = width // 2, channels // 2
    x = x[:, :, : width2 * 2, : channels2 * 2]
    x = x.reshape((batch, height, width2, 2, channels2, 2))
    return jnp.mean(x, axis=(3, 5))


class ActorCritic(nnx.Module):
    def __init__(
        self,
        num_actions: int,
        in_shape,
        activation: str = "tanh",
        *,
        rngs: nnx.Rngs,
    ):
        assert activation in ["relu", "tanh"]
        height, width, in_channels = (int(x) for x in in_shape)
        self.num_actions = num_actions
        self.activation = activation

        self.conv = nnx.Conv(in_channels, 32, kernel_size=(2, 2), padding="SAME", rngs=rngs)
        # Haiku's avg_pool applies its window to the width and channel axes.
        flat_features = height * (width // 2) * (32 // 2)
        self.fc = nnx.Linear(flat_features, 64, rngs=rngs)

        self.actor1 = nnx.Linear(64, 64, rngs=rngs)
        self.actor2 = nnx.Linear(64, 64, rngs=rngs)
        self.actor_out = nnx.Linear(64, num_actions, rngs=rngs)

        self.critic1 = nnx.Linear(64, 64, rngs=rngs)
        self.critic2 = nnx.Linear(64, 64, rngs=rngs)
        self.critic_out = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, x):
        x = x.astype(jnp.float32)
        activation = jax.nn.relu if self.activation == "relu" else jax.nn.tanh

        x = nnx.relu(self.conv(x))
        x = _avg_pool2x2(x)
        x = x.reshape((x.shape[0], -1))
        x = nnx.relu(self.fc(x))

        actor_mean = activation(self.actor1(x))
        actor_mean = activation(self.actor2(actor_mean))
        actor_mean = self.actor_out(actor_mean)

        critic = activation(self.critic1(x))
        critic = activation(self.critic2(critic))
        critic = self.critic_out(critic)

        return actor_mean, jnp.squeeze(critic, axis=-1)


def make_model(seed: int = 0) -> ActorCritic:
    return ActorCritic(
        env.num_actions,
        in_shape=env.observation_shape,
        activation="tanh",
        rngs=nnx.Rngs(seed),
    )


def make_optimizer(model: ActorCritic) -> nnx.Optimizer:
    return nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.lr, eps=1e-5),
        ),
        wrt=nnx.Param,
    )


# A tiny categorical distribution built on JAX, replacing the previous
# `distrax.Categorical` dependency.
def _log_softmax(logits):
    return jax.nn.log_softmax(logits, axis=-1)


def _sample(rng, logits):
    return jax.random.categorical(rng, logits, axis=-1)


def _log_prob(logits, action):
    log_probs = _log_softmax(logits)
    return jnp.take_along_axis(log_probs, action[..., None], axis=-1)[..., 0]


def _entropy(logits):
    log_probs = _log_softmax(logits)
    return -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


def _calculate_gae(traj_batch, last_val):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )
        delta = reward + args.gamma * next_value * (1 - done) - value
        gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value


def make_collect_fn():
    @nnx.jit
    def collect(model, runner_state):
        def _env_step(runner_state, unused):
            env_state, last_obs, rng = runner_state
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            logits, value = model(last_obs)
            action = _sample(_rng, logits)
            log_prob = _log_prob(logits, action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            keys = jax.random.split(_rng, env_state.observation.shape[0])
            env_state = jax.vmap(auto_reset(env.step, env.init))(env_state, action, keys)
            transition = Transition(
                env_state.terminated,
                action,
                value,
                jnp.squeeze(env_state.rewards),
                log_prob,
                last_obs,
            )
            runner_state = (env_state, env_state.observation, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, args.num_steps)
        env_state, last_obs, rng = runner_state
        _, last_val = model(last_obs)
        return (env_state, last_obs, rng), traj_batch, last_val

    return collect


def make_update_fn():
    def _loss_fn(model, traj_batch, gae, targets):
        # RERUN NETWORK
        logits, value = model(traj_batch.obs)
        log_prob = _log_prob(logits, traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-args.clip_eps, args.clip_eps)
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - args.clip_eps,
                1.0 + args.clip_eps,
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = _entropy(logits).mean()

        total_loss = loss_actor + args.vf_coef * value_loss - args.ent_coef * entropy
        return total_loss, (value_loss, loss_actor, entropy)

    @nnx.jit
    def update(model, optimizer, runner_state, traj_batch, last_val):
        advantages, targets = _calculate_gae(traj_batch, last_val)

        rng = runner_state[2]
        rng, _rng = jax.random.split(rng)
        batch_size = args.minibatch_size * num_minibatches
        assert (
            batch_size == args.num_steps * args.num_envs
        ), "batch size must be equal to number of steps * number of envs"
        permutation = jax.random.permutation(_rng, batch_size)
        batch = (traj_batch, advantages, targets)
        batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree.map(
            lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
            shuffled_batch,
        )

        graphdef, state = nnx.split(model)
        opt_graphdef, opt_state = nnx.split(optimizer)

        def _update_minbatch(carry, batch_info):
            state, opt_state = carry
            model = nnx.merge(graphdef, state)
            optimizer = nnx.merge(opt_graphdef, opt_state)
            traj_batch, advantages, targets = batch_info
            (total_loss, _), grads = nnx.value_and_grad(_loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)(
                model, traj_batch, advantages, targets
            )
            optimizer.update(model, grads)
            return (nnx.state(model), nnx.state(optimizer)), total_loss

        (state, opt_state), loss_info = jax.lax.scan(_update_minbatch, (state, opt_state), minibatches)
        nnx.update(model, state)
        nnx.update(optimizer, opt_state)

        env_state, last_obs, rng = runner_state
        return (env_state, last_obs, rng), loss_info

    return update


@nnx.jit
def evaluate(model, rng_key):
    step_fn = jax.vmap(env.step)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, args.num_eval_envs)
    state = jax.vmap(env.init)(subkeys)
    R = jnp.zeros_like(state.rewards)

    def cond_fn(tup):
        state, _, _ = tup
        return ~state.terminated.all()

    def loop_fn(tup):
        state, R, rng_key = tup
        logits, _ = model(state.observation)
        rng_key, _rng = jax.random.split(rng_key)
        action = _sample(_rng, logits)
        rng_key, _rng = jax.random.split(rng_key)
        keys = jax.random.split(_rng, state.observation.shape[0])
        state = step_fn(state, action, keys)
        return state, R + state.rewards, rng_key

    state, R, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, R, rng_key))
    return R.mean()


def train(rng):
    tt = 0
    st = time.time()
    # INIT NETWORK
    model = make_model(args.seed)
    optimizer = make_optimizer(model)

    collect = make_collect_fn()
    update = make_update_fn()

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, args.num_envs)
    env_state = jax.jit(jax.vmap(env.init))(reset_rng)

    rng, _rng = jax.random.split(rng)
    runner_state = (env_state, env_state.observation, _rng)

    # warm up
    model.eval()
    _, _, _ = collect(model, runner_state)

    steps = 0

    # initial evaluation
    et = time.time()  # exclude evaluation time
    tt += et - st
    rng, _rng = jax.random.split(rng)
    model.eval()
    eval_R = evaluate(model, _rng)
    log = {"sec": tt, f"{args.env_name}/eval_R": float(eval_R), "steps": steps}
    print(log)
    wandb.log(log)
    st = time.time()

    for i in range(num_updates):
        model.eval()
        runner_state, traj_batch, last_val = collect(model, runner_state)
        model.train()
        runner_state, loss_info = update(model, optimizer, runner_state, traj_batch, last_val)
        steps += args.num_envs * args.num_steps

        # evaluation
        et = time.time()  # exclude evaluation time
        tt += et - st
        rng, _rng = jax.random.split(rng)
        model.eval()
        eval_R = evaluate(model, _rng)
        log = {"sec": tt, f"{args.env_name}/eval_R": float(eval_R), "steps": steps}
        print(log)
        wandb.log(log)
        st = time.time()

    return model, optimizer


if __name__ == "__main__":
    wandb.init(project=args.wandb_project, config=args.model_dump())
    rng = jax.random.PRNGKey(args.seed)
    model, optimizer = train(rng)
    if args.save_model:
        with open(f"{args.env_name}-seed={args.seed}.ckpt", "wb") as f:
            pickle.dump(
                {
                    "model": jax.device_get(nnx.state(model)),
                    "opt_state": jax.device_get(nnx.state(optimizer)),
                },
                f,
            )
