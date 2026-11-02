import os
import pickle
from typing import Literal, Sequence

import jax
import jax.numpy as jnp

from pgx._src.utils import _download

BaselineModelId = Literal[
    "animal_shogi_v0",
    "gardner_chess_v0",
    "go_9x9_v0",
    "hex_v0",
    "othello_v0",
    "minatar-asterix_v0",
    "minatar-breakout_v0",
    "minatar-freeway_v0",
    "minatar-seaquest_v0",
    "minatar-space_invaders_v0",
]


def make_baseline_model(model_id: BaselineModelId, download_dir: str = "baselines"):
    if model_id in (
        "animal_shogi_v0",
        "gardner_chess_v0",
        "go_9x9_v0",
        "hex_v0",
        "othello_v0",
    ):
        return _make_az_baseline_model(model_id, download_dir)
    elif model_id in (
        "minatar-asterix_v0",
        "minatar-breakout_v0",
        "minatar-freeway_v0",
        "minatar-seaquest_v0",
        "minatar-space_invaders_v0",
    ):
        return _make_minatar_baseline_model(model_id, download_dir)
    else:
        # game for which not baseline exists should not be fatal,
        # so let caller deal with it
        return None


def _get_observation_shape(model_id: BaselineModelId) -> Sequence[int]:
    # NNX modules initialize eagerly, so the input shape has to be known before
    # the pretrained weights can be loaded into them.
    from pgx.core import make

    env_id = model_id[: -len("_v0")]
    return make(env_id).observation_shape  # type: ignore[arg-type]


def _make_az_baseline_model(model_id: BaselineModelId, download_dir: str = "baselines"):
    from flax import nnx

    model_args, model_params, model_state = _load_baseline_model(model_id, download_dir)
    model = _create_az_model_v0(
        **model_args,
        in_shape=_get_observation_shape(model_id),
        rngs=nnx.Rngs(0),
    )
    _convert_az_params(
        model,
        model_params,
        model_state,
        resnet_v2=model_args.get("resnet_v2", True),
    )
    model.eval()

    def apply(obs):
        return model(obs)

    return apply


def _make_minatar_baseline_model(model_id: BaselineModelId, download_dir: str = "baselines"):
    from flax import nnx

    model_args, model_params, model_state = _load_baseline_model(model_id, download_dir)
    del model_state
    model = _create_minatar_model(
        **model_args,
        in_shape=_get_observation_shape(model_id),
        rngs=nnx.Rngs(0),
    )
    _convert_minatar_params(model, model_params)
    model.eval()

    def apply(obs):
        return model(obs)

    return apply


def _load_baseline_model(baseline_model: BaselineModelId, basedir: str = "baselines"):
    os.makedirs(basedir, exist_ok=True)

    # download baseline model if not exists
    filename = os.path.join(basedir, baseline_model + ".ckpt")
    if not os.path.exists(filename):
        url = _get_download_url(baseline_model)
        _download(url, filename)

    with open(filename, "rb") as f:
        d = pickle.load(f)

    return d["args"], d["params"], d["state"]


def _get_download_url(baseline_model: BaselineModelId) -> str:
    urls = {
        "animal_shogi_v0": "https://drive.google.com/uc?id=1HpP5GLf9b6zkJL8FKUFfKS8Zycs-gzZg",
        "gardner_chess_v0": "https://drive.google.com/uc?id=1RUdrxhYseG-FliskVdemNYYM5YYmfwU7",
        "go_9x9_v0": "https://drive.google.com/uc?id=1hXMicBALW3WU43NquDoX4zthY4-KjiVu",
        "hex_v0": "https://drive.google.com/uc?id=11qpLAT4_0NgPrKRcJCPE7RdN92VP8Ws3",
        "othello_v0": "https://drive.google.com/uc?id=1mY40mWoPuYCOrlfMQk_6DPGEFaQcvNAM",
        "minatar-asterix_v0": "https://drive.google.com/uc?id=1ohUxhZTYQCwyH-WJRH_Ma9BV3M1WoY0N",
        "minatar-breakout_v0": "https://drive.google.com/uc?id=1ED1-p3Gmi4PZEH3hF-9NZzPyNkiPCnvT",
        "minatar-freeway_v0": "https://drive.google.com/uc?id=1rbnJGxlzAWkt5DtF7tiYwoNkSqk0l2kD",
        "minatar-seaquest_v0": "https://drive.google.com/uc?id=1740nIi00Z8fQWRbA-52GiSGkW7rcqM8o",
        "minatar-space_invaders_v0": "https://drive.google.com/uc?id=1I7kJ8GEhY9K3rAFnbnYtlI5KQusFReq9",
    }
    assert baseline_model in urls
    return urls[baseline_model]


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------


def _get(tree, key):
    """Read a Haiku parameter entry from a flat (``a/b``) or nested tree."""
    if key in tree:
        return tree[key]
    node = tree
    for part in key.split("/"):
        node = node[part]
    return node


def _set_linear(layer, params):
    layer.kernel[...] = params["w"]
    layer.bias[...] = params["b"]


def _set_conv(layer, params):
    layer.kernel[...] = params["w"]
    layer.bias[...] = params["b"]


def _set_batch_norm(layer, params, mean_state, var_state):
    layer.scale[...] = params["scale"].reshape((-1,))
    layer.bias[...] = params["offset"].reshape((-1,))
    layer.mean[...] = mean_state["average"].reshape((-1,))
    layer.var[...] = var_state["average"].reshape((-1,))


def _convert_az_params(model, params, state, resnet_v2: bool = True):
    """Copy a Haiku AlphaZero parameter tree into an NNX module."""
    _set_conv(model.conv_init, _get(params, "az_net/conv2_d"))

    if resnet_v2:
        _set_batch_norm(
            model.bn_final,
            _get(params, "az_net/batch_norm"),
            _get(state, "az_net/batch_norm/~/mean_ema"),
            _get(state, "az_net/batch_norm/~/var_ema"),
        )
    else:
        _set_batch_norm(
            model.bn_init,
            _get(params, "az_net/batch_norm"),
            _get(state, "az_net/batch_norm/~/mean_ema"),
            _get(state, "az_net/batch_norm/~/var_ema"),
        )

    for i, block in enumerate(model.blocks):
        prefix = f"az_net/block_{i}"
        _set_conv(block.conv1, _get(params, f"{prefix}/conv2_d"))
        _set_batch_norm(
            block.bn1,
            _get(params, f"{prefix}/batch_norm"),
            _get(state, f"{prefix}/batch_norm/~/mean_ema"),
            _get(state, f"{prefix}/batch_norm/~/var_ema"),
        )
        _set_conv(block.conv2, _get(params, f"{prefix}/conv2_d_1"))
        _set_batch_norm(
            block.bn2,
            _get(params, f"{prefix}/batch_norm_1"),
            _get(state, f"{prefix}/batch_norm_1/~/mean_ema"),
            _get(state, f"{prefix}/batch_norm_1/~/var_ema"),
        )

    # Policy head
    _set_conv(model.policy_conv, _get(params, "az_net/conv2_d_1"))
    _set_batch_norm(
        model.policy_bn,
        _get(params, "az_net/batch_norm_1"),
        _get(state, "az_net/batch_norm_1/~/mean_ema"),
        _get(state, "az_net/batch_norm_1/~/var_ema"),
    )
    _set_linear(model.policy_linear, _get(params, "az_net/linear"))

    # Value head
    _set_conv(model.value_conv, _get(params, "az_net/conv2_d_2"))
    _set_batch_norm(
        model.value_bn,
        _get(params, "az_net/batch_norm_2"),
        _get(state, "az_net/batch_norm_2/~/mean_ema"),
        _get(state, "az_net/batch_norm_2/~/var_ema"),
    )
    _set_linear(model.value_linear1, _get(params, "az_net/linear_1"))
    _set_linear(model.value_linear2, _get(params, "az_net/linear_2"))


def _convert_minatar_params(model, params):
    """Copy a Haiku MinAtar PPO parameter tree into an NNX module."""
    _set_conv(model.conv, _get(params, "actor_critic/conv2_d"))
    _set_linear(model.fc, _get(params, "actor_critic/linear"))
    _set_linear(model.actor1, _get(params, "actor_critic/linear_1"))
    _set_linear(model.actor2, _get(params, "actor_critic/linear_2"))
    _set_linear(model.actor_out, _get(params, "actor_critic/linear_3"))
    _set_linear(model.critic1, _get(params, "actor_critic/linear_4"))
    _set_linear(model.critic2, _get(params, "actor_critic/linear_5"))
    _set_linear(model.critic_out, _get(params, "actor_critic/linear_6"))


def _avg_pool2x2(x):
    """Match ``hk.avg_pool(x, (2, 2), (2, 2), "VALID")`` for NHWC inputs.

    Haiku prepends leading ones to the window shape when it has fewer entries
    than the input rank, so the window is applied to the width and channel
    axes (not height and width). The pretrained MinAtar baselines were trained
    with this behaviour, so it is preserved here.
    """
    batch, height, width, channels = x.shape
    width2, channels2 = width // 2, channels // 2
    x = x[:, :, : width2 * 2, : channels2 * 2]
    x = x.reshape((batch, height, width2, 2, channels2, 2))
    return jnp.mean(x, axis=(3, 5))


# ---------------------------------------------------------------------------
# Models (kept in this module so `flax` remains an optional dependency)
# ---------------------------------------------------------------------------


def _create_az_model_v0(
    num_actions,
    num_channels: int = 128,
    num_layers: int = 6,
    resnet_v2: bool = True,
    *,
    in_shape: Sequence[int],
    rngs=None,
):
    from flax import nnx

    class BlockV1(nnx.Module):
        def __init__(self, num_channels: int, *, rngs: nnx.Rngs):
            self.conv1 = nnx.Conv(num_channels, num_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
            self.bn1 = nnx.BatchNorm(num_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)
            self.conv2 = nnx.Conv(num_channels, num_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
            self.bn2 = nnx.BatchNorm(num_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

        def __call__(self, x):
            residual = x
            x = nnx.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return nnx.relu(x + residual)

    class BlockV2(nnx.Module):
        def __init__(self, num_channels: int, *, rngs: nnx.Rngs):
            self.bn1 = nnx.BatchNorm(num_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)
            self.conv1 = nnx.Conv(num_channels, num_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
            self.bn2 = nnx.BatchNorm(num_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)
            self.conv2 = nnx.Conv(num_channels, num_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)

        def __call__(self, x):
            residual = x
            x = nnx.relu(self.bn1(x))
            x = self.conv1(x)
            x = nnx.relu(self.bn2(x))
            x = self.conv2(x)
            return x + residual

    class AZNet(nnx.Module):
        """AlphaZero NN architecture."""

        def __init__(
            self,
            num_actions,
            num_channels: int,
            num_layers: int,
            resnet_v2: bool,
            in_shape: Sequence[int],
            *,
            rngs: nnx.Rngs,
        ):
            height, width, in_channels = (int(x) for x in in_shape)
            self.num_actions = num_actions
            self.num_channels = num_channels
            self.num_layers = num_layers
            self.resnet_v2 = resnet_v2
            self.in_shape = (height, width, in_channels)

            self.conv_init = nnx.Conv(in_channels, num_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
            if not resnet_v2:
                self.bn_init = nnx.BatchNorm(num_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

            block_cls = BlockV2 if resnet_v2 else BlockV1
            self.blocks = nnx.List([block_cls(num_channels, rngs=rngs) for _ in range(num_layers)])

            if resnet_v2:
                self.bn_final = nnx.BatchNorm(num_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

            # Policy head
            self.policy_conv = nnx.Conv(num_channels, 2, kernel_size=(1, 1), padding="SAME", rngs=rngs)
            self.policy_bn = nnx.BatchNorm(2, momentum=0.9, epsilon=1e-5, rngs=rngs)
            self.policy_linear = nnx.Linear(2 * height * width, num_actions, rngs=rngs)

            # Value head
            self.value_conv = nnx.Conv(num_channels, 1, kernel_size=(1, 1), padding="SAME", rngs=rngs)
            self.value_bn = nnx.BatchNorm(1, momentum=0.9, epsilon=1e-5, rngs=rngs)
            self.value_linear1 = nnx.Linear(height * width, num_channels, rngs=rngs)
            self.value_linear2 = nnx.Linear(num_channels, 1, rngs=rngs)

        def __call__(self, x):
            x = x.astype(jnp.float32)
            x = self.conv_init(x)

            if not self.resnet_v2:
                x = nnx.relu(self.bn_init(x))

            for block in self.blocks:
                x = block(x)

            if self.resnet_v2:
                x = nnx.relu(self.bn_final(x))

            # Policy head
            logits = self.policy_conv(x)
            logits = nnx.relu(self.policy_bn(logits))
            logits = logits.reshape((logits.shape[0], -1))
            logits = self.policy_linear(logits)

            # Value head
            value = self.value_conv(x)
            value = nnx.relu(self.value_bn(value))
            value = value.reshape((value.shape[0], -1))
            value = nnx.relu(self.value_linear1(value))
            value = self.value_linear2(value)
            value = jnp.tanh(value)
            value = value.reshape((-1,))

            return logits, value

    if rngs is None:
        rngs = nnx.Rngs(0)
    return AZNet(
        num_actions,
        num_channels,
        num_layers,
        resnet_v2,
        in_shape,
        rngs=rngs,
    )


def _create_minatar_model(
    num_actions,
    activation: str = "tanh",
    *,
    in_shape: Sequence[int],
    rngs=None,
):
    from flax import nnx

    assert activation in ["relu", "tanh"]

    class ActorCritic(nnx.Module):
        def __init__(self, num_actions, activation, in_shape, *, rngs: nnx.Rngs):
            height, width, in_channels = (int(x) for x in in_shape)
            self.num_actions = num_actions
            self.activation = activation

            self.conv = nnx.Conv(in_channels, 32, kernel_size=(2, 2), padding="SAME", rngs=rngs)
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

    if rngs is None:
        rngs = nnx.Rngs(0)
    return ActorCritic(num_actions, activation, in_shape, rngs=rngs)
