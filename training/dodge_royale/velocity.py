"""A convolutional danger field, read by a fixed controller.

Adapted from DodgeAI ``dodge/velocity.py`` at revision e32f222
(``VelocityFlowV2Extractor``), which is a separate project and is neither
imported nor modified. The architecture is ported; no weights are. Arena
scale, hazards and the movement model all differ, so a Royale policy trains
from scratch.

The network's only output is a scalar danger per cell per time slice. It never
chooses a direction. Choosing is arithmetic: read the field along each of the
nine candidate paths, then blend those nine headings in proportion to how safe
each one came back. Nothing about the controller is learned, so all the
learning pressure lands on the field.

The nine are candidates, not choices. The blend of them is a vector, so the
agent travels on any heading -- and a small change in the field moves that
heading a little, where picking the best of nine would have swung the whole
action the instant two readings crossed.

**What Royale changes, and why.**

*The paths are given, not assumed.* DodgeAI holds a fixed ``offsets`` buffer:
where each action takes a player starting from rest. Royale's player carries
real momentum -- about four times its current velocity, up to ten pixels -- so
a rest-start path is misplaced by roughly two and a half cells, which at 4px
cells is most of a hitbox. The gym predicts the nine paths from the player's
actual position and velocity and ships them in the observation, so the
controller reads them rather than assuming them, and the sampled point is
where the player will really be.

*The window is player-centred.* DodgeAI observes the whole arena and has to be
told where in it the player is. Royale's window travels with the player, so
there are no absolute coordinates to feed the critic, and the player is always
at the centre. That makes the path coordinates and the sampling coordinates
the same numbers -- see :func:`sample_points`.

*The grid is 64x64 of 4px cells, not 32x32.* An ordinary Royale enemy is four
to seven pixels across, so coarser cells would average away the detail that
decides a dodge.
"""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F

from .protocol import Layout

__all__ = [
    "NO_THREAT_DISTANCE",
    "SAMPLE_DECAY",
    "VelocityFlowRoyaleExtractor",
    "decode_observation",
    "sample_points",
    "sense_features",
]

#: Slope of the leaky rectifier, as in the architecture this is adapted from.
NEGATIVE_SLOPE = 0.01

#: Discount applied to a horizon's sample. Gentle enough that the far samples
#: still carry weight: at 0.93 the 108-frame sample would be worth 0.0004 and
#: the horizon would be decorative.
SAMPLE_DECAY = 0.985

#: Initial logit sharpness. Low enough that the first rollouts are near
#: uniform -- committing hard to a randomly initialised field would be
#: committing to noise -- and learnable, so the policy sharpens as the field
#: becomes worth trusting.
INITIAL_TEMPERATURE = 0.5

TRUNK_CHANNELS = 32
CONTEXT_CHANNELS = 16
CONTEXT_GRID = 4
VALUE_FEATURES = 64
VALUE_SUMMARY_CHANNELS = 8
VALUE_SUMMARY_GRID = 4
VALUE_SUMMARY_DIM = VALUE_SUMMARY_CHANNELS * VALUE_SUMMARY_GRID**2

#: Channels whose occupancy can end an episode. `blast-phase` is excluded: it
#: says where a blast is in its life, not that a cell is occupied, and a
#: shrinking blast carries a negative phase that would read as "safe".
LETHAL_CHANNELS = ("normal-enemy", "kamikaze", "blast")
VELOCITY_CHANNEL_NAMES = ("velocity-x", "velocity-y")

SENSE_FEATURES = 6

#: Distance reported when no lethal cell is in the window, in the same units
#: the real distances use (cells, divided by the grid edge). Finite and
#: deliberately unreachable: the window's own diagonal is about 0.71 of a grid
#: edge, so nothing real can reach 4.0, and the critic sees one consistent
#: number for "nothing here" rather than an infinity that would poison any
#: layer it reached.
NO_THREAT_DISTANCE = 4.0


def activation() -> nn.Module:
    return nn.LeakyReLU(NEGATIVE_SLOPE)


def _sample_weights(horizons: tuple[int, ...], decay: float = SAMPLE_DECAY) -> np.ndarray:
    weights = np.array([decay**frame for frame in horizons], dtype=np.float32)
    return weights / weights.sum()


def decode_observation(
    observations: torch.Tensor, layout: Layout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a flat observation into the player, the grid, and the paths.

    Sections come from the layout rather than from constants, so a layout this
    build does not expect fails in `Layout.validate` rather than here, silently
    reading the wrong slice.
    """
    batch = observations.shape[0]
    player = layout.player_section.slice(observations)
    grids = layout.grid_section.slice(observations).reshape(
        batch, len(layout.channels), layout.grid, layout.grid
    )
    paths = layout.path_section.slice(observations).reshape(
        batch, len(layout.actions), len(layout.horizons), 2
    )
    return player, grids, paths


def sample_points(paths: torch.Tensor, layout: Layout) -> torch.Tensor:
    """Turn stored paths into `grid_sample` coordinates.

    A stored path is a displacement from the player, in window pixels divided
    by `path_scale`. The player sits at the window's centre, so the pixel point
    is ``centre + path * path_scale``, and `grid_sample` wants that normalised
    to [-1, 1] across the window with ``align_corners=False``.

    Written as a scale and an offset rather than literally, because the literal
    form loses precision where it matters most. ``2 * (128 + p * 128) / 256 - 1``
    computes ``1 + p`` and then subtracts one, so a small ``p`` -- a short path,
    which is the common case -- is rounded into the gap beside 1.0 and comes
    back with its low bits gone. Folding the constants first keeps every path
    exact:

        scale  = 2 * path_scale / window_pixels
        offset = 2 * centre / window_pixels - 1

    For the shipped layout the centre is half the window and `path_scale` is
    its half width, so this is a multiply by one and an add of zero -- the
    stored path already *is* its coordinate. That is a property of this layout
    and not a law, which is why the conversion is computed from the layout
    rather than assumed: a window that stopped being player-centred would
    otherwise sample cells the player never reaches, while every tensor stayed
    the right shape.
    """
    scale = 2.0 * layout.path_scale / layout.window_pixels
    offset = 2.0 * (layout.window_pixels / 2.0) / layout.window_pixels - 1.0
    if scale == 1.0 and offset == 0.0:
        # Exactly the identity for this layout; skip the arithmetic so no
        # rounding is introduced where none is needed.
        return paths
    return paths * scale + offset


def sense_features(
    player: torch.Tensor, grids: torch.Tensor, layout: Layout
) -> torch.Tensor:
    """Nearest-threat geometry, and how fast it is actually closing.

    Six numbers: the player's own velocity, the offset to the nearest lethal
    cell, its distance, and the closing speed. Every term is exact rather than
    inferred -- velocity comes from the observation, and closing is the real
    relative speed along the line to the threat, not the change in distance
    between two quantised frames.

    Measured from the geometric centre of the window, because that is where
    the player is. There is no centre cell: the player sits on the corner
    shared by cells 31 and 32, so a cell's displacement is ``col + 0.5 - 32``.
    """
    batch = grids.shape[0]
    device, dtype = grids.device, grids.dtype
    edge = float(layout.grid)
    half = edge / 2.0

    lethal = [layout.channels.index(name) for name in LETHAL_CHANNELS]
    velocity_planes = [layout.channels.index(name) for name in VELOCITY_CHANNEL_NAMES]

    occupied = grids[:, lethal].amax(dim=1) > 0

    # Cell centres relative to the player, in cells.
    centres = torch.arange(layout.grid, device=device, dtype=dtype) + 0.5 - half
    dx = centres.reshape(1, 1, -1)
    dy = centres.reshape(1, -1, 1)
    squared = dx * dx + dy * dy

    # Anything unoccupied is pushed past every real distance, so the minimum
    # is over threats only and an empty window is recognisable afterwards.
    beyond = float(layout.grid * layout.grid * 16)
    masked = torch.where(occupied, squared.expand(batch, -1, -1), squared.new_full((), beyond))
    best, index = masked.reshape(batch, -1).min(dim=1)
    empty = best >= beyond

    distance = torch.where(
        empty,
        best.new_full((), NO_THREAT_DISTANCE),
        torch.sqrt(best.clamp(max=beyond - 1.0)) / edge,
    )
    column = (index % layout.grid).to(dtype)
    row = torch.div(index, layout.grid, rounding_mode="floor").to(dtype)
    nearest_x = column + 0.5 - half
    nearest_y = row + 0.5 - half

    # Closing speed: relative velocity projected onto the line to the threat,
    # positive when the gap is shrinking. The encoded velocities are world
    # velocities, so the player's own is subtracted here.
    player_velocity = player[:, :2]
    rows = torch.arange(batch, device=device)
    flat = grids.reshape(batch, grids.shape[1], -1)
    threat_x = flat[rows, velocity_planes[0], index]
    threat_y = flat[rows, velocity_planes[1], index]
    relative_x = threat_x - player_velocity[:, 0]
    relative_y = threat_y - player_velocity[:, 1]
    # A threat sitting exactly on the player has no direction to close along.
    # Clamping rather than dividing by zero keeps that case a finite zero
    # instead of a NaN that would spread through every later layer.
    length = torch.sqrt(nearest_x * nearest_x + nearest_y * nearest_y).clamp(min=1e-6)
    closing = -(relative_x * nearest_x + relative_y * nearest_y) / length

    zero = nearest_x.new_zeros(())
    nearest_x = torch.where(empty, zero, nearest_x / edge)
    nearest_y = torch.where(empty, zero, nearest_y / edge)
    closing = torch.where(empty, zero, closing)
    return torch.stack(
        (
            player_velocity[:, 0],
            player_velocity[:, 1],
            nearest_x,
            nearest_y,
            distance,
            closing,
        ),
        dim=1,
    )


class _Trunk(nn.Module):
    """A stem at full resolution, a body at half, interpolated back.

    The stem stays at 64x64 so the first layer still sees every 4px cell --
    which is most of an enemy -- while the three heavy convolutions run at
    32x32, where the multiply-accumulate count is a quarter. Bilinear rather
    than nearest on the way back: a step at a block edge is not a fact about
    danger, and the controller samples across those edges.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), activation())
        self.body = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            activation(),
            # Dilated, to reach across the window without another downsample.
            # At half resolution its reach doubles in arena terms: 11 cells of
            # 8px rather than of 4px.
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            activation(),
            nn.Conv2d(64, TRUNK_CHANNELS, 3, padding=1),
            activation(),
        )

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        stem = self.stem(grids)
        body = self.body(F.avg_pool2d(stem, 2))
        return F.interpolate(body, size=stem.shape[-2:], mode="bilinear", align_corners=False)


class VelocityFlowRoyaleExtractor(BaseFeaturesExtractor):
    """Produce a danger field, then read it along each action's real path.

    Returns ``len(actions) + VALUE_FEATURES`` features: the negated danger per
    candidate path, which the policy blends into a heading, followed by the
    critic's own.
    """

    def __init__(self, observation_space, layout: Layout | dict) -> None:
        self.layout = layout if isinstance(layout, Layout) else Layout.from_json(layout)
        actions = len(self.layout.actions)
        horizons = len(self.layout.horizons)
        super().__init__(observation_space, features_dim=actions + VALUE_FEATURES)

        if observation_space.shape != (self.layout.observation_values,):
            raise ValueError(
                f"this layout describes {self.layout.observation_values} values, but the "
                f"observation space is {observation_space.shape}"
            )

        self.field_net = _Trunk(len(self.layout.channels))
        # Window-wide context, folded back in at full resolution. The trunk's
        # receptive field cannot span the window, so "which side is safe" was
        # not a question a single field cell could ask. Pooling to 4x4 and
        # broadcasting back gives every cell the whole window's summary while
        # the field stays fully convolutional.
        self.context_net = nn.Sequential(
            nn.Conv2d(TRUNK_CHANNELS, 32, 3, padding=1),
            activation(),
            nn.Conv2d(32, CONTEXT_CHANNELS, 3, padding=1),
            activation(),
        )
        # One output per horizon rather than one field: slice k answers "how
        # dangerous is this cell at frame horizons[k]".
        self.field_head = nn.Conv2d(TRUNK_CHANNELS + CONTEXT_CHANNELS, horizons, 1)

        # 1x1, so the summary is a channel projection of each cell rather than
        # another receptive field -- the trunk has already done that work.
        self.value_pool = nn.Conv2d(TRUNK_CHANNELS, VALUE_SUMMARY_CHANNELS, 1)
        coordinate_inputs = self.layout.player_section.length + SENSE_FEATURES
        self.value_net = nn.Sequential(
            nn.Linear(TRUNK_CHANNELS + VALUE_SUMMARY_DIM + coordinate_inputs, VALUE_FEATURES),
            activation(),
            nn.Linear(VALUE_FEATURES, VALUE_FEATURES),
            activation(),
        )
        self.register_buffer(
            "sample_weights", torch.as_tensor(_sample_weights(self.layout.horizons))
        )

    # -- pieces, exposed so they can be tested and drawn --

    def field(self, observations: torch.Tensor) -> torch.Tensor:
        """Danger per cell per horizon: ``(batch, horizons, grid, grid)``.

        Slice 0 is the nearest future, and is what a debug overlay draws.
        """
        _, grids, _ = decode_observation(observations, self.layout)
        return self._field_from_trunk(self.field_net(grids))

    def _field_from_trunk(self, trunk: torch.Tensor) -> torch.Tensor:
        context = self.context_net(F.adaptive_avg_pool2d(trunk, CONTEXT_GRID))
        # Bilinear, not nearest: nearest would hold each of the 4x4 values flat
        # across a 16x16 block, arriving as sixteen hard-edged tiles that swamp
        # the trunk's per-cell detail. A global summary should be a smooth
        # low-frequency surface.
        context = F.interpolate(
            context, size=trunk.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.field_head(torch.cat((trunk, context), dim=1))

    def value_summary(self, trunk: torch.Tensor) -> torch.Tensor:
        """A coarse map of the window for the critic: ``(batch, 8 * 4 * 4)``.

        A mean over the whole field cannot say *where* anything is, and "am I
        about to die" is almost entirely a question of where.
        """
        return F.adaptive_avg_pool2d(self.value_pool(trunk), VALUE_SUMMARY_GRID).flatten(1)

    def danger(self, observations: torch.Tensor) -> torch.Tensor:
        """Weighted danger along each action's path: ``(batch, actions)``."""
        _, grids, paths = decode_observation(observations, self.layout)
        return self._danger_from_field(self._field_from_trunk(self.field_net(grids)), paths)

    def _danger_from_field(self, field: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        sampled = F.grid_sample(
            field,
            sample_points(paths, self.layout),
            mode="bilinear",
            # The paths are deliberately unclipped and may leave the window.
            # Border padding reads the edge rather than zero, so a path that
            # runs off the side is judged by the last thing actually seen
            # instead of by a fabricated safe region.
            padding_mode="border",
            align_corners=False,
        )
        # Each sample must be read from the slice for its own arrival time:
        # where the player will be at frame t, judged by the field predicted
        # for frame t. That pairing is the diagonal, and any other entry would
        # ask whether a cell is safe at a time the player is not in it. It is
        # what lets "lethal now, clear in two seconds" be expressed at all.
        paired = sampled.permute(0, 2, 1, 3).diagonal(dim1=2, dim2=3)
        return (paired * self.sample_weights).sum(dim=-1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        player, grids, paths = decode_observation(observations, self.layout)
        trunk = self.field_net(grids)
        field = self._field_from_trunk(trunk)
        danger = self._danger_from_field(field, paths)

        senses = sense_features(player, grids, self.layout)
        pooled = trunk.mean(dim=(2, 3))
        value = self.value_net(
            torch.cat((pooled, self.value_summary(trunk), player, senses), dim=1)
        )
        # Negated: the controller prefers low danger, and these become the
        # weights the nine candidate headings are blended by.
        return torch.cat((-danger, value), dim=1)
