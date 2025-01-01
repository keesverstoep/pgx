import jax
import jax.numpy as jnp
from pgx.awari import Awari

env = Awari()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)

# TODO TODO

def test_init():
    key = jax.random.PRNGKey(0)
    _, key = jax.random.split(key)  # due to API update
    _, key = jax.random.split(key)  # due to API update
    state = init(key=key)
    assert state.current_player == 0


def test_step():
    key = jax.random.PRNGKey(0)
    state = init(key)
    state = step(state, 4)
    # fmt: off
    expected = jnp.int32([
        5, 5, 5, 4, 4, 4,
        4, 4, 4, 4, 0, 5,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0])
    # fmt:on
    assert jnp.all(state._x.board == expected)


def test_terminated():
    # wipe out
    key = jax.random.PRNGKey(0)
    _, key = jax.random.split(key)  # due to API update
    _, key = jax.random.split(key)  # due to API update
    state = init(key)
    for i in [0, 1, 2, 3, 4, 5]:
        state = step(state, i)
        assert not state.terminated
    assert (state.rewards == jnp.float32([0.0, 0.0])).all()
    # TODO: add full game and check termination
    # assert state.terminated


def test_legal_action():
    # cannot put
    key = jax.random.PRNGKey(0)
    _, key = jax.random.split(key)  # due to API update
    _, key = jax.random.split(key)  # due to API update
    state = init(key)
    assert state.current_player == 0
    for i in [0, 1, 2, 3, 4, 5]:
        state = step(state, i)
    # still actions available for resulting board
    assert state.legal_action_mask[:6].any()
    # but no pass
    assert ~state.legal_action_mask[6]
    # TODO: check exact action mask

    # check (invalid) pass:
    state = step(state, 6)
    assert state.terminated


def test_observe():
    key = jax.random.PRNGKey(0)
    _, key = jax.random.split(key)  # due to API update
    _, key = jax.random.split(key)  # due to API update
    state = init(key)

    obs = observe(state, state.current_player)
    # print("test_observe 0 obs:", obs)
    assert obs.shape == (2, 6, 4)

    state = step(state, 5)
    # print("test_observe 1 state:", state)
    obs = observe(state, state.current_player)
    # print("test_observe 1 obs", obs)
    obs_expected1 = jnp.array([
       [[5, 0, 0, 0],
        [5, 0, 0, 0],
        [5, 0, 0, 0],
        [5, 0, 0, 0],
        [4, 0, 0, 0],
        [4, 0, 0, 0]],

       [[0, 4, 0, 0],
        [0, 4, 0, 0],
        [0, 4, 0, 0],
        [0, 4, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 0, 0]]
    ])

    obs_other = observe(state, (1 - state.current_player))
    obs_expected_1_other = jnp.array([
       [[4, 0, 0, 0],
        [4, 0, 0, 0],
        [4, 0, 0, 0],
        [4, 0, 0, 0],
        [4, 0, 0, 0],
        [0, 0, 0, 0]],

       [[0, 5, 0, 0],
        [0, 5, 0, 0],
        [0, 5, 0, 0],
        [0, 5, 0, 0],
        [0, 4, 0, 0],
        [0, 4, 0, 0]]
    ])
    # print("test_observe 1 obs_expected", obs_expected_1_other)
    assert (obs_other == obs_expected_1_other).all()

    state = step(state, 3)
    # print("test_observe 2 state:", state)
    obs = observe(state, state.current_player)
    obs_expected_2 = jnp.array([
       [[5, 0, 0, 0],
        [5, 0, 0, 0],
        [5, 0, 0, 0],
        [4, 0, 0, 0],
        [4, 0, 0, 0],
        [0, 0, 0, 0]],

       [[0, 5, 0, 0],
        [0, 5, 0, 0],
        [0, 5, 0, 0],
        [0, 0, 0, 0],
        [0, 5, 0, 0],
        [0, 5, 0, 0]]
    ])
    # print("test_observe 2 obs:", obs)
    # print("test_observe 2 obs_expected:", obs_expected_2)
    assert (obs == obs_expected_2).all()

    obs_other = observe(state, (1 - state.current_player))
    obs_expected_2_other = jnp.array([
       [[5, 0, 0, 0],
        [5, 0, 0, 0],
        [5, 0, 0, 0],
        [0, 0, 0, 0],
        [5, 0, 0, 0],
        [5, 0, 0, 0]],

       [[0, 5, 0, 0],
        [0, 5, 0, 0],
        [0, 5, 0, 0],
        [0, 4, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 0, 0]]
    ])
    # print("test_observe 2 obs other:", obs_other)
    # print("test_observe 2 obs_expected_other:", obs_expected_2)
    assert (obs_other == obs_expected_2_other).all()

def test_api():
    import pgx
    env = pgx.make("awari")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)
