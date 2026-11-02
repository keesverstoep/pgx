# AlphaZero network implemented with Flax NNX.
#
# The architecture follows the ResNet used in AlphaZero and the original
# Haiku implementation this file replaces:
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx


class BlockV1(nnx.Module):
    """ResNet v1 block: conv -> BN -> ReLU -> conv -> BN -> add -> ReLU."""

    def __init__(self, num_channels: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            num_channels,
            num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(num_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.conv2 = nnx.Conv(
            num_channels,
            num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(num_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = nnx.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return nnx.relu(x + residual)


class BlockV2(nnx.Module):
    """Pre-activation ResNet v2 block: BN -> ReLU -> conv -> BN -> ReLU -> conv."""

    def __init__(self, num_channels: int, *, rngs: nnx.Rngs):
        self.bn1 = nnx.BatchNorm(num_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.conv1 = nnx.Conv(
            num_channels,
            num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(num_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.conv2 = nnx.Conv(
            num_channels,
            num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = nnx.relu(self.bn1(x))
        x = self.conv1(x)
        x = nnx.relu(self.bn2(x))
        x = self.conv2(x)
        return x + residual


class AZNet(nnx.Module):
    """AlphaZero policy/value network.

    Args:
        num_actions: Size of the discrete action space.
        in_shape: Shape of a single observation, e.g. ``env.observation_shape``.
            The last dimension is the number of input channels.
        num_channels: Number of channels used by the residual tower.
        num_blocks: Number of residual blocks.
        resnet_v2: Use pre-activation (v2) residual blocks.
        rngs: NNX random number streams used for eager initialization.
    """

    def __init__(
        self,
        num_actions: int,
        in_shape: Sequence[int],
        num_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        if len(in_shape) != 3:
            raise ValueError(f"`in_shape` must be (height, width, channels), got {tuple(in_shape)}")
        height, width, in_channels = (int(x) for x in in_shape)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        self.in_shape: Tuple[int, int, int] = (height, width, in_channels)

        self.conv_init = nnx.Conv(
            in_channels,
            num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )
        if not resnet_v2:
            self.bn_init = nnx.BatchNorm(num_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

        block_cls = BlockV2 if resnet_v2 else BlockV1
        self.blocks = nnx.List([block_cls(num_channels, rngs=rngs) for _ in range(num_blocks)])

        if resnet_v2:
            self.bn_final = nnx.BatchNorm(num_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

        # Policy head. The spatial dimensions are preserved by the residual
        # tower, so the flattened feature size is known at construction time.
        self.policy_conv = nnx.Conv(num_channels, 2, kernel_size=(1, 1), padding="SAME", rngs=rngs)
        self.policy_bn = nnx.BatchNorm(2, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.policy_linear = nnx.Linear(2 * height * width, num_actions, rngs=rngs)

        # Value head.
        self.value_conv = nnx.Conv(num_channels, 1, kernel_size=(1, 1), padding="SAME", rngs=rngs)
        self.value_bn = nnx.BatchNorm(1, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.value_linear1 = nnx.Linear(height * width, num_channels, rngs=rngs)
        self.value_linear2 = nnx.Linear(num_channels, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Compute policy logits and value for a batch of observations.

        Call ``model.train()`` or ``model.eval()`` to switch batch-norm
        behaviour. This mirrors Haiku's ``is_training`` argument without
        threading it through every layer.
        """
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
