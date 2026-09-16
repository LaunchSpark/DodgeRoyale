"""The extractor, tested on what it computes rather than on what shape it is.

A shape-correct controller can still rank the actions wrongly: sample the
wrong horizon's slice, flip Y, offset the points twice, and every tensor is
still the right size. So most of these plant a known field, aim a known path
into it, and assert which action comes out on top.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from dodge_royale.policies import (
    ARCHITECTURES,
    ROYALE_ARCHITECTURE,
    VelocityFlowRoyalePolicy,
    _FieldLogits,
    checkpoint_architecture,
    checkpoint_layout,
    policy_kwargs_for,
    require_loadable,
)
from dodge_royale.protocol import Layout, ProtocolError
from dodge_royale.velocity import (
    NO_THREAT_DISTANCE,
    SAMPLE_DECAY,
    VALUE_FEATURES,
    VelocityFlowRoyaleExtractor,
    decode_observation,
    sample_points,
    sense_features,
)
from dodge_royale.vec_env import observation_space


@pytest.fixture(scope="session")
def layout(manifest) -> Layout:
    return Layout.from_json(manifest["layout"])


@pytest.fixture
def extractor(layout) -> VelocityFlowRoyaleExtractor:
    torch.manual_seed(0)
    return VelocityFlowRoyaleExtractor(observation_space(layout), layout)


def blank(layout: Layout, batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, layout.observation_values)


def put_grid(observation: torch.Tensor, layout: Layout, channel: str, grid: torch.Tensor) -> None:
    """Write one 64x64 channel into a flat observation."""
    index = layout.channels.index(channel)
    cells = layout.grid**2
    start = layout.grid_section.offset + index * cells
    observation[..., start : start + cells] = grid.reshape(-1)


def put_path(
    observation: torch.Tensor, layout: Layout, action: int, horizon: int, dx: float, dy: float
) -> None:
    base = layout.path_section.offset + (action * len(layout.horizons) + horizon) * 2
    observation[..., base] = dx
    observation[..., base + 1] = dy


def set_all_paths(observation: torch.Tensor, layout: Layout, action: int, dx: float, dy: float):
    for horizon in range(len(layout.horizons)):
        put_path(observation, layout, action, horizon, dx, dy)


# --- decoding ------------------------------------------------------------


def test_the_observation_splits_into_the_parts_the_layout_names(layout):
    observation = blank(layout, batch=3)
    player, grids, paths = decode_observation(observation, layout)
    assert player.shape == (3, 2)
    assert grids.shape == (3, len(layout.channels), layout.grid, layout.grid)
    assert paths.shape == (3, len(layout.actions), len(layout.horizons), 2)


def test_a_channel_written_flat_comes_back_as_a_square(layout):
    observation = blank(layout)
    marked = torch.zeros(layout.grid, layout.grid)
    marked[7, 13] = 1.0  # row 7, column 13
    put_grid(observation, layout, "kamikaze", marked)

    _, grids, _ = decode_observation(observation, layout)
    channel = grids[0, layout.channels.index("kamikaze")]
    assert channel[7, 13] == 1.0
    assert channel.sum() == 1.0, "row-major, and nothing smeared"


# --- the coordinate identity ---------------------------------------------


def test_stored_paths_are_already_grid_sample_coordinates(layout):
    """The equivalence that catches a double offset or a flipped Y.

    ``points = centre + path * 128`` normalised over a 256px window with
    align_corners=False is the path itself. If either the centring or the
    scale drifts, this stops holding and the controller samples somewhere the
    player will never be -- while every tensor stays the right shape.
    """
    paths = torch.tensor([[[[0.0, 0.0], [0.5, -0.25], [-1.0, 1.0], [2.0, -3.0]]]])
    centre = layout.window_pixels / 2.0
    points = centre + paths * layout.path_scale
    normalised = 2.0 * points / layout.window_pixels - 1.0

    assert torch.allclose(normalised, paths, atol=0.0), "the identity is exact"
    assert torch.equal(sample_points(paths, layout), normalised)


def test_the_window_centre_is_the_corner_between_the_middle_cells(layout):
    """There is no centre cell, which is what align_corners=False expects."""
    assert layout.grid % 2 == 0
    assert layout.window_pixels / 2.0 == layout.path_scale
    assert layout.grid * layout.cell_pixels == layout.window_pixels


# --- the controller ------------------------------------------------------


def plant_field(extractor, values: torch.Tensor) -> None:
    """Replace the trunk so the field is exactly `values`.

    `values` is (horizons, grid, grid). Monkeypatching the field rather than
    training one is the only way to know what the controller *should* choose,
    which is the thing being tested.
    """
    extractor._field_from_trunk = lambda trunk: values.unsqueeze(0).expand(
        trunk.shape[0], -1, -1, -1
    )


def test_the_controller_avoids_the_action_whose_path_lands_in_danger(extractor, layout):
    horizons = len(layout.horizons)
    field = torch.zeros(horizons, layout.grid, layout.grid)
    # A hot region on the right-hand side, at every horizon.
    field[:, :, layout.grid // 2 + 8 :] = 10.0
    plant_field(extractor, field)

    observation = blank(layout)
    set_all_paths(observation, layout, action=2, dx=0.75, dy=0.0)  # right, into it
    set_all_paths(observation, layout, action=1, dx=-0.75, dy=0.0)  # left, away

    danger = extractor.danger(observation)[0]
    assert danger[2] > danger[1], "the path into the hot region is the dangerous one"

    features = extractor(observation)[0]
    # Features are negated danger, so the safe action scores higher.
    assert features[1] > features[2]


def test_y_is_screen_down_not_world_up(extractor, layout):
    """A flipped Y would send every vertical action to the wrong half."""
    horizons = len(layout.horizons)
    field = torch.zeros(horizons, layout.grid, layout.grid)
    # Hot along the bottom rows, which is positive dy in the observation.
    field[:, layout.grid // 2 + 8 :, :] = 10.0
    plant_field(extractor, field)

    observation = blank(layout)
    set_all_paths(observation, layout, action=4, dx=0.0, dy=0.75)  # down
    set_all_paths(observation, layout, action=3, dx=0.0, dy=-0.75)  # up

    danger = extractor.danger(observation)[0]
    assert danger[4] > danger[3], "positive dy reads the lower rows"


def test_each_horizon_reads_its_own_slice(extractor, layout):
    """The diagonal pairing, which nothing else in the shape would catch.

    Slice k is hot in a place only horizon k's path reaches. An extractor that
    sampled one field for every horizon, or paired the slices off by one,
    picks up either nothing or the wrong slice's danger.
    """
    horizons = len(layout.horizons)
    field = torch.zeros(horizons, layout.grid, layout.grid)
    # A distinct hot column per slice, walking rightwards.
    columns = [layout.grid // 2 + 4 * (index + 1) for index in range(horizons)]
    for index, column in enumerate(columns):
        field[index, :, column] = 100.0
    plant_field(extractor, field)

    centre = layout.grid / 2.0
    for target in range(horizons):
        observation = blank(layout)
        # Every horizon of action 0 aims at slice `target`'s hot column.
        aim = (columns[target] + 0.5 - centre) / centre
        for horizon in range(horizons):
            put_path(observation, layout, 0, horizon, aim, 0.0)
        danger = extractor.danger(observation)[0, 0]

        # Only the horizon whose own slice is hot there contributes, weighted.
        weights = extractor.sample_weights
        expected = 100.0 * float(weights[target])
        assert danger == pytest.approx(expected, rel=0.05), (
            f"horizon {target} must read slice {target}, not another"
        )


def test_the_horizon_weights_decay_and_sum_to_one(extractor, layout):
    weights = extractor.sample_weights
    assert float(weights.sum()) == pytest.approx(1.0)
    assert all(weights[i] > weights[i + 1] for i in range(len(weights) - 1))
    raw = np.array([SAMPLE_DECAY**frame for frame in layout.horizons], dtype=np.float32)
    assert np.allclose(weights.numpy(), raw / raw.sum())


def test_a_path_leaving_the_window_reads_the_border_not_a_safe_zero(extractor, layout):
    """Paths are deliberately unclipped; zero padding would invent safety."""
    horizons = len(layout.horizons)
    field = torch.full((horizons, layout.grid, layout.grid), 5.0)
    plant_field(extractor, field)

    observation = blank(layout)
    set_all_paths(observation, layout, action=0, dx=3.0, dy=3.0)  # far outside
    danger = extractor.danger(observation)[0, 0]
    assert danger == pytest.approx(5.0, rel=1e-4), "the border value, not zero"


def test_identical_paths_give_identical_danger(extractor, layout):
    horizons = len(layout.horizons)
    torch.manual_seed(1)
    plant_field(extractor, torch.randn(horizons, layout.grid, layout.grid))

    observation = blank(layout)
    for action in range(len(layout.actions)):
        set_all_paths(observation, layout, action, dx=0.1, dy=-0.2)
    danger = extractor.danger(observation)[0]
    assert torch.allclose(danger, danger[0].expand_as(danger), atol=1e-6)


def test_different_velocities_produce_different_paths_and_different_rankings(extractor, layout):
    """The reason paths are shipped rather than assumed.

    Two observations with the same field but different paths -- which is what
    different momentum produces -- must rank the actions differently. An
    extractor holding a fixed rest-start offset table would score them alike.
    """
    horizons = len(layout.horizons)
    field = torch.zeros(horizons, layout.grid, layout.grid)
    field[:, :, layout.grid // 2 + 8 :] = 10.0
    plant_field(extractor, field)

    drifting_right = blank(layout)
    set_all_paths(drifting_right, layout, action=0, dx=0.75, dy=0.0)
    drifting_left = blank(layout)
    set_all_paths(drifting_left, layout, action=0, dx=-0.75, dy=0.0)

    right = extractor.danger(drifting_right)[0, 0]
    left = extractor.danger(drifting_left)[0, 0]
    assert right > left, "idle is dangerous when momentum carries you into a threat"


# --- senses --------------------------------------------------------------


def test_senses_find_the_nearest_lethal_cell(layout):
    observation = blank(layout)
    grid = torch.zeros(layout.grid, layout.grid)
    grid[32, 40] = 1.0  # eight cells right of centre, on the centre row
    put_grid(observation, layout, "normal-enemy", grid)

    player, grids, _ = decode_observation(observation, layout)
    senses = sense_features(player, grids, layout)[0]
    _, _, nearest_x, nearest_y, distance, _ = senses

    assert nearest_x > 0, "to the right"
    assert nearest_y == pytest.approx(0.5 / layout.grid, abs=1e-6), "on the centre row"
    assert distance == pytest.approx(
        np.hypot(40 + 0.5 - 32, 32 + 0.5 - 32) / layout.grid, rel=1e-4
    )


def test_an_empty_window_reports_the_no_threat_sentinel(layout):
    """Finite, and out of reach of any real distance."""
    observation = blank(layout)
    player, grids, _ = decode_observation(observation, layout)
    senses = sense_features(player, grids, layout)[0]
    assert float(senses[4]) == pytest.approx(NO_THREAT_DISTANCE)
    assert senses[2] == 0.0 and senses[3] == 0.0
    assert senses[5] == 0.0, "nothing is closing"
    assert torch.isfinite(senses).all()

    # The window's own diagonal cannot reach the sentinel.
    furthest = np.hypot(layout.grid / 2, layout.grid / 2) / layout.grid
    assert furthest < NO_THREAT_DISTANCE


def test_blast_phase_is_not_treated_as_occupancy(layout):
    """A shrinking blast's negative phase would otherwise read as safe."""
    observation = blank(layout)
    phase = torch.zeros(layout.grid, layout.grid)
    phase[10, 10] = -0.8
    put_grid(observation, layout, "blast-phase", phase)

    player, grids, _ = decode_observation(observation, layout)
    senses = sense_features(player, grids, layout)[0]
    assert float(senses[4]) == pytest.approx(NO_THREAT_DISTANCE), "phase is not a hazard"


def test_a_threat_on_the_player_does_not_produce_a_nan(layout):
    """Zero distance has no direction to close along."""
    observation = blank(layout)
    grid = torch.zeros(layout.grid, layout.grid)
    # As close to the centre as a cell gets: the corner cells around it.
    grid[31, 31] = 1.0
    grid[31, 32] = 1.0
    grid[32, 31] = 1.0
    grid[32, 32] = 1.0
    put_grid(observation, layout, "blast", grid)
    velocity = torch.zeros(layout.grid, layout.grid)
    velocity[31, 31] = 0.5
    put_grid(observation, layout, "velocity-x", velocity)

    player, grids, _ = decode_observation(observation, layout)
    senses = sense_features(player, grids, layout)
    assert torch.isfinite(senses).all(), senses


def test_closing_speed_is_relative_to_the_player(layout):
    """The encoded velocity is the world's; the sense is the closing rate."""
    observation = blank(layout)
    grid = torch.zeros(layout.grid, layout.grid)
    grid[32, 40] = 1.0
    put_grid(observation, layout, "normal-enemy", grid)
    approaching = torch.zeros(layout.grid, layout.grid)
    approaching[32, 40] = -0.5  # moving left, towards the player
    put_grid(observation, layout, "velocity-x", approaching)

    player, grids, _ = decode_observation(observation, layout)
    closing = sense_features(player, grids, layout)[0, 5]
    assert closing > 0, "a shrinking gap is positive closing"

    # Now let the player chase it at the same speed: the gap stops shrinking.
    observation[..., 0] = -0.5
    player, grids, _ = decode_observation(observation, layout)
    assert sense_features(player, grids, layout)[0, 5] == pytest.approx(0.0, abs=1e-5)


# --- the whole extractor -------------------------------------------------


def test_the_extractor_returns_seventy_three_features(extractor, layout):
    observation = blank(layout, batch=4)
    features = extractor(observation)
    assert features.shape == (4, len(layout.actions) + VALUE_FEATURES) == (4, 73)
    assert extractor.features_dim == 73


def test_the_field_has_one_slice_per_horizon(extractor, layout):
    field = extractor.field(blank(layout, batch=2))
    assert field.shape == (2, len(layout.horizons), layout.grid, layout.grid)


def test_the_critic_summary_is_eight_channels_on_a_four_by_four_grid(extractor, layout):
    _, grids, _ = decode_observation(blank(layout, batch=2), layout)
    trunk = extractor.field_net(grids)
    assert trunk.shape == (2, 32, layout.grid, layout.grid), "interpolated back to full size"
    assert extractor.value_summary(trunk).shape == (2, 8 * 4 * 4)


def test_the_critic_takes_one_hundred_and_sixty_eight_inputs(extractor):
    first = extractor.value_net[0]
    assert first.in_features == 32 + 128 + 2 + 6 == 168


def test_features_and_gradients_are_finite_on_real_observations(manifest, extractor, layout):
    """Real bytes, not zeros: a NaN that only appears on live data is the
    kind that survives every synthetic test."""
    import io

    from conftest import FIXTURES
    from dodge_royale.protocol import decode_step

    raw = (FIXTURES / "scenes.bin").read_bytes()
    batch = decode_step(io.BytesIO(raw), 3, manifest["observation_values"])
    observation = torch.as_tensor(batch.observations)

    features = extractor(observation)
    assert torch.isfinite(features).all()
    features.sum().backward()
    for name, parameter in extractor.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_a_layout_that_does_not_match_the_space_is_refused(layout):
    wrong = spaces.Box(low=-1.0, high=1.0, shape=(123,), dtype=np.float32)
    with pytest.raises(ValueError, match="observation space"):
        VelocityFlowRoyaleExtractor(wrong, layout)


# --- logits --------------------------------------------------------------


def test_logits_are_standardised_so_only_the_field_shape_matters():
    head = _FieldLogits(actions=9)
    # Comfortably above EPSILON, so the spread clamp is not what is being
    # measured here -- that case has its own test below.
    small = torch.randn(4, 73)
    large = small.clone()
    large[:, :9] *= 1000.0

    from_small = head(small)
    from_large = head(large)
    assert torch.allclose(from_small, from_large, atol=1e-4), (
        "scaling the field must not sharpen the policy"
    )


def test_logits_survive_nine_identical_dangers():
    """Zero spread would be a division by zero without the clamp."""
    head = _FieldLogits(actions=9)
    latent = torch.zeros(2, 73)
    logits = head(latent)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits, torch.zeros_like(logits))


def test_the_initial_temperature_keeps_the_first_policy_near_uniform():
    head = _FieldLogits(actions=9)
    torch.manual_seed(0)
    latent = torch.randn(1, 73)
    probabilities = torch.softmax(head(latent), dim=1).detach()
    assert float(probabilities.max()) < 0.35, "committing to a random field is committing to noise"


# --- policy and checkpoints ----------------------------------------------


def build_policy(layout: Layout):
    kwargs = policy_kwargs_for(layout)
    return VelocityFlowRoyalePolicy(
        observation_space(layout),
        spaces.Discrete(len(layout.actions)),
        lr_schedule=lambda _: 3e-4,
        **kwargs,
    )


def test_the_policy_uses_the_field_for_its_logits(layout):
    policy = build_policy(layout)
    assert isinstance(policy.action_net, _FieldLogits)
    # An empty pi net: the logits are the field's own reading, and a hidden
    # layer between them would be the learned head this design does without.
    assert policy.mlp_extractor.latent_dim_pi == 73


def test_the_policy_produces_finite_actions_and_values(layout):
    policy = build_policy(layout)
    observation = torch.zeros(3, layout.observation_values)
    actions, values, log_prob = policy(observation)
    assert actions.shape == (3,)
    assert values.shape == (3, 1)
    assert torch.isfinite(values).all()
    assert torch.isfinite(log_prob).all()
    assert int(actions.max()) < len(layout.actions)


def test_the_architecture_table_names_royale_and_carries_its_hyperparameters():
    entry = ARCHITECTURES[ROYALE_ARCHITECTURE]
    assert entry.policy_class is VelocityFlowRoyalePolicy
    # Converted from four-frame to one-frame decisions, preserving time scales.
    assert entry.gamma == pytest.approx(0.99 ** (1 / 4))
    assert entry.gae_lambda == pytest.approx(0.95 ** (1 / 4))


def test_every_table_entry_is_data_not_a_callable():
    """A dict with one callable among plain values is a trap for a caller
    that reads them uniformly, which is exactly what the training CLI will
    do."""
    for entry in ARCHITECTURES.values():
        assert isinstance(entry.gamma, float)
        assert isinstance(entry.gae_lambda, float)
        assert not callable(entry.gamma) and not callable(entry.gae_lambda)


def test_both_the_fresh_and_resumed_paths_take_the_discounts_from_the_table(layout):
    """Neither path may hardcode a discount beside a table that holds one."""
    entry = ARCHITECTURES[ROYALE_ARCHITECTURE]

    fresh = entry.ppo_kwargs(layout)
    assert fresh["gamma"] == entry.gamma
    assert fresh["gae_lambda"] == entry.gae_lambda
    assert fresh["policy"] is VelocityFlowRoyalePolicy
    assert isinstance(fresh["policy_kwargs"], dict), "kwargs, not the function that builds them"

    resumed = entry.resume_kwargs()["custom_objects"]
    assert resumed["gamma"] == entry.gamma
    assert resumed["gae_lambda"] == entry.gae_lambda
    # The value PPO defaults to, and the one the plan warns against leaving in.
    assert entry.gae_lambda != 0.95


def test_a_resumed_checkpoint_actually_takes_the_tables_discounts(tmp_path, layout):
    """SB3 restores what was saved unless told otherwise, so this checks the
    override reaches the loaded model rather than only the kwargs dict."""
    from stable_baselines3 import PPO

    entry = ARCHITECTURES[ROYALE_ARCHITECTURE]
    stale = PPO(
        **entry.ppo_kwargs(layout),
        env=_DummyEnv(layout),
        n_steps=8,
        batch_size=8,
        device="cpu",
    )
    stale.gae_lambda = 0.95  # as if saved before the table was corrected
    stale.gamma = 0.99
    path = tmp_path / "stale.zip"
    stale.save(path)

    assert PPO.load(path, device="cpu").gae_lambda == pytest.approx(0.95)
    resumed = PPO.load(path, device="cpu", **entry.resume_kwargs())
    assert resumed.gae_lambda == pytest.approx(entry.gae_lambda)
    assert resumed.gamma == pytest.approx(entry.gamma)


def test_a_saved_checkpoint_reloads_with_the_same_outputs(tmp_path, layout, manifest):
    """Save and reload must be a no-op for the policy's behaviour."""
    import io

    from stable_baselines3 import PPO

    from conftest import FIXTURES
    from dodge_royale.protocol import decode_step

    raw = (FIXTURES / "scenes.bin").read_bytes()
    decoded = decode_step(io.BytesIO(raw), 3, manifest["observation_values"])
    observation = torch.as_tensor(decoded.observations)

    torch.manual_seed(0)
    model = PPO(
        VelocityFlowRoyalePolicy,
        _DummyEnv(layout),
        policy_kwargs=policy_kwargs_for(layout),
        n_steps=8,
        batch_size=8,
        device="cpu",
    )
    model.policy.set_training_mode(False)
    # Move the temperature off its initial value, so a reload that silently
    # rebuilt _FieldLogits from scratch would not coincidentally match.
    with torch.no_grad():
        model.policy.action_net.log_temperature.fill_(0.37)
    with torch.no_grad():
        before_values = model.policy.predict_values(observation)
        before_logits = model.policy.get_distribution(observation).distribution.logits

    path = tmp_path / "royale.zip"
    model.save(path)
    reloaded = PPO.load(path, device="cpu")
    reloaded.policy.set_training_mode(False)
    with torch.no_grad():
        after_values = reloaded.policy.predict_values(observation)
        after_logits = reloaded.policy.get_distribution(observation).distribution.logits

    # The critic and the actor. Logits rather than sampled actions: sampling is
    # stochastic and log_prob depends on which action was drawn, so neither
    # would match across a reload without seeding, and comparing them would be
    # testing the seed rather than the restore.
    assert torch.allclose(before_values, after_values, atol=1e-6)
    assert torch.allclose(before_logits, after_logits, atol=1e-6)
    assert reloaded.policy.action_net.log_temperature.item() == pytest.approx(0.37)
    assert checkpoint_layout(path) == layout


def test_a_checkpoint_trained_on_another_layout_is_refused(tmp_path, layout):
    """Identical length, different meaning: the dangerous case.

    Two layouts of equal size whose channels are ordered differently would
    load, run, and point every trained filter at a different thing.
    """
    from stable_baselines3 import PPO

    model = PPO(
        VelocityFlowRoyalePolicy,
        _DummyEnv(layout),
        policy_kwargs=policy_kwargs_for(layout),
        n_steps=8,
        batch_size=8,
        device="cpu",
    )
    path = tmp_path / "royale.zip"
    model.save(path)

    raw = layout.as_dict()
    raw["channels"] = [raw["channels"][1], raw["channels"][0], *raw["channels"][2:]]
    served = Layout.from_json(raw)
    assert served.observation_values == layout.observation_values

    require_loadable(path, layout)  # the matching one is fine
    with pytest.raises(ProtocolError, match="not the one this policy was trained"):
        require_loadable(path, served)


def test_a_checkpoint_without_a_layout_is_refused(tmp_path, layout):
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", _DummyEnv(layout), n_steps=8, batch_size=8, device="cpu")
    path = tmp_path / "foreign.zip"
    model.save(path)
    with pytest.raises(ProtocolError):
        require_loadable(path, layout)


class _DummyEnv(VecEnv):
    """The smallest thing PPO will accept, so a policy can be built to save.

    A real `VecEnv` subclass rather than a duck type: PPO patches anything
    that is not one through a Gymnasium shim, which rejects a stand-in.
    """

    def __init__(self, layout: Layout) -> None:
        self._layout = layout
        super().__init__(
            num_envs=1,
            observation_space=observation_space(layout),
            action_space=spaces.Discrete(len(layout.actions)),
        )

    def get_attr(self, attr_name, indices=None):
        if attr_name != "render_mode":
            raise AttributeError(attr_name)
        return [None]

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(attr_name)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise AttributeError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]

    def reset(self):
        return np.zeros((1, self._layout.observation_values), dtype=np.float32)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        return (
            np.zeros((1, self._layout.observation_values), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=bool),
            [{}],
        )

    def close(self):
        pass


# --- the trunk itself ----------------------------------------------------


def test_the_body_actually_runs_at_half_resolution(extractor, layout):
    """The one structural claim the controller tests cannot make.

    Those monkeypatch the field away, so an implementation that dropped the
    pool and the interpolate entirely would pass every one of them and still
    produce a (batch, 32, 64, 64) trunk. Hooks are the only way to see where
    the work happened: full resolution into the stem, half out of it into the
    body, full again after the interpolate.
    """
    seen: dict[str, tuple[int, ...]] = {}
    trunk = extractor.field_net

    # These must return None. A forward hook that returns anything replaces
    # the module's output with it, so a lambda whose body is a tuple of two
    # assignments quietly hands the next layer a tuple.
    def watch_stem(_module, _inputs, output) -> None:
        seen["stem_out"] = tuple(output.shape)

    def watch_body(_module, inputs, output) -> None:
        seen["body_in"] = tuple(inputs[0].shape)
        seen["body_out"] = tuple(output.shape)

    handles = [
        trunk.stem.register_forward_hook(watch_stem),
        trunk.body.register_forward_hook(watch_body),
    ]
    try:
        _, grids, _ = decode_observation(blank(layout, batch=2), layout)
        output = trunk(grids)
    finally:
        for handle in handles:
            handle.remove()

    full, half = layout.grid, layout.grid // 2
    assert seen["stem_out"][-2:] == (full, full), "the stem sees every 4px cell"
    assert seen["body_in"][-2:] == (half, half), "the heavy convolutions run at half"
    assert seen["body_out"][-2:] == (half, half)
    assert tuple(output.shape) == (2, 32, full, full), "and it is interpolated back"


def test_the_trunk_is_upsampled_bilinearly_not_by_repetition(extractor, layout):
    """Nearest would hold each body cell flat across a 2x2 block.

    That matters because the controller samples across those edges: a step at
    a block boundary is an artefact of the upsample, not a fact about danger.
    """
    torch.manual_seed(3)
    grids = torch.randn(1, len(layout.channels), layout.grid, layout.grid)
    output = extractor.field_net(grids)[0, 0]

    # Compare each cell with its 2x2 block partner. Under nearest they are
    # identical everywhere; under bilinear they differ almost everywhere.
    blocks = output.reshape(layout.grid // 2, 2, layout.grid // 2, 2)
    spread = (blocks.amax(dim=(1, 3)) - blocks.amin(dim=(1, 3))).detach()
    assert float(spread.mean()) > 1e-6, "a nearest upsample would make every block flat"


def test_the_stem_sees_a_single_cell_that_the_body_would_blur_away(extractor, layout):
    """The reason the stem stays at full resolution.

    An ordinary Royale enemy is four to seven pixels -- one or two cells -- so
    a stem at half resolution would average a hitbox with its empty neighbour
    before anything looked at it.
    """
    grids = torch.zeros(1, len(layout.channels), layout.grid, layout.grid)
    grids[0, layout.channels.index("normal-enemy"), 20, 21] = 1.0
    stem = extractor.field_net.stem(grids)
    # The lone cell has to move the stem somewhere, at full resolution.
    assert stem.shape[-2:] == (layout.grid, layout.grid)
    empty = extractor.field_net.stem(torch.zeros_like(grids))
    assert not torch.allclose(stem, empty), "one occupied cell must reach the stem"


def test_sample_points_converts_rather_than_assuming_the_identity(layout):
    """It must be right for a layout it was not written against.

    The identity holds only because the window is player-centred and
    `path_scale` is its half width. A layout where that stops being true has
    to come out converted, not unchanged -- otherwise the function is a
    comment and the test beside it is the only thing standing between a
    changed layout and a controller sampling the wrong cells.
    """
    raw = layout.as_dict()
    raw["path_scale"] = layout.window_pixels / 4.0  # half the half width
    halved = Layout.from_json(raw)

    paths = torch.tensor([[[[0.5, -0.25], [1.0, 0.0]]]])
    converted = sample_points(paths, halved)

    assert torch.allclose(converted, paths * 0.5), "the scale must follow the layout"
    assert not torch.equal(converted, paths), "and it is no longer the identity"
    # The shipped layout is still exactly the identity, with no arithmetic.
    assert torch.equal(sample_points(paths, layout), paths)


# --- the recorded architecture name --------------------------------------


def test_a_saved_checkpoint_records_the_architecture_that_wrote_it(tmp_path, layout):
    """The name has to survive into the zip, not just exist in memory.

    SB3 splats `policy_kwargs` into the policy constructor, so a name recorded
    there only works because the policy accepts and swallows it. If that ever
    stops being true this fails at construction, and if the key stops being
    saved it fails here -- either way, before the fallback can quietly cover
    for it.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.save_util import load_from_zip_file

    model = PPO(
        **ARCHITECTURES[ROYALE_ARCHITECTURE].ppo_kwargs(layout),
        env=_DummyEnv(layout),
        n_steps=8,
        batch_size=8,
        device="cpu",
    )
    assert model.policy.architecture == ROYALE_ARCHITECTURE

    path = tmp_path / "named.zip"
    model.save(path)
    data, _, _ = load_from_zip_file(path, load_data=True, device="cpu")
    assert data["policy_kwargs"]["architecture"] == ROYALE_ARCHITECTURE, (
        "the name must be in the saved kwargs, not only on the live object"
    )
    assert checkpoint_architecture(path) == ROYALE_ARCHITECTURE
    assert PPO.load(path, device="cpu").policy.architecture == ROYALE_ARCHITECTURE


def test_a_checkpoint_naming_another_architecture_is_refused(tmp_path, layout):
    """Proves the recorded name is what decides, not the extractor class.

    This checkpoint uses the Royale extractor and the Royale policy, so the
    class-identity fallback would call it ours. Only the recorded name says
    otherwise, so if that branch were dead this would be accepted.
    """
    from stable_baselines3 import PPO

    kwargs = ARCHITECTURES[ROYALE_ARCHITECTURE].ppo_kwargs(layout)
    kwargs["policy_kwargs"] = {**kwargs["policy_kwargs"], "architecture": "velocity-flow-v2"}
    model = PPO(**kwargs, env=_DummyEnv(layout), n_steps=8, batch_size=8, device="cpu")
    path = tmp_path / "foreign-name.zip"
    model.save(path)

    assert checkpoint_architecture(path) == "velocity-flow-v2"
    with pytest.raises(ProtocolError, match="only builds"):
        require_loadable(path, layout)
