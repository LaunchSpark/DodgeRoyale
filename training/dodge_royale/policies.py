"""The Royale policy, and the checkpoint compatibility rules around it.

Adapted from DodgeAI ``dodge/policies.py`` and ``dodge/velocity.py`` at
revision e32f222 (``_FieldLogits``, ``VelocityFlowPolicy``). DodgeAI is a
separate project; nothing here imports it, and a Royale checkpoint never needs
it to load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from .protocol import ACTION_DIRECTIONS, Layout, ProtocolError
from .velocity import INITIAL_TEMPERATURE, VALUE_FEATURES, VelocityFlowRoyaleExtractor

__all__ = [
    "ARCHITECTURES",
    "LOG_STD_INIT",
    "SUPERSEDED_ARCHITECTURES",
    "ROYALE",
    "Architecture",
    "ROYALE_ARCHITECTURE",
    "VelocityFlowRoyalePolicy",
    "checkpoint_architecture",
    "checkpoint_layout",
    "policy_kwargs_for",
    "require_loadable",
    "unit_directions",
]

#: The only architecture this trainer builds. Named in checkpoints, so a run
#: can say what it was.
#:
#: The `-direction` suffix marks the action space, not a rewrite. A checkpoint
#: from before it holds a nine-way categorical head, and its weights cannot be
#: loaded into a policy that emits a heading -- the layout and the extractor are
#: unchanged, so nothing else about the checkpoint would give that away. Without
#: the name, an old checkpoint passes every compatibility gate and fails deep
#: inside torch with a missing state-dict key, which says nothing about why.
ROYALE_ARCHITECTURE = "velocity-flow-royale-direction"

#: What this trainer used to build, kept only so a checkpoint from then can be
#: recognised and refused for the right reason.
SUPERSEDED_ARCHITECTURES = {
    "velocity-flow-royale": (
        "it was trained on the nine-way action space, before actions became "
        "directions; its policy head cannot be loaded into this one"
    ),
}

#: Starting spread of the Gaussian around the blended heading, as a log.
#:
#: SB3 defaults to zero, meaning a standard deviation of one -- as large as the
#: unit heading it is added to, so the sampled direction would be very nearly
#: noise and the field would have almost no say in where the agent went. At
#: -1.0 the deviation is about 0.37, which perturbs a unit heading by roughly
#: twenty degrees: enough to explore either side of a choice, small enough that
#: the choice survives. PPO tunes it from there.
LOG_STD_INIT = -1.0


def unit_directions(layout: Layout) -> torch.Tensor:
    """The nine candidate paths as unit headings: ``(actions, 2)``.

    Normalised, because the blend below is a weighted sum of these. The
    simulation writes a diagonal as ``(1, 1)``, and summing that unchanged
    would give the corners half again the pull of the axes -- a policy that
    slightly preferred up-right would drift further than one that equally
    preferred right.
    """
    rows = []
    for name in layout.actions:
        try:
            x, y = ACTION_DIRECTIONS[name]
        except KeyError as error:
            raise ProtocolError(
                f"this policy has no heading for the action {name!r}"
            ) from error
        length = float(np.hypot(x, y))
        rows.append((x / length, y / length) if length else (0.0, 0.0))
    return torch.tensor(rows, dtype=torch.float32)


class _FieldDirection(nn.Module):
    """Turn path danger into a heading, with a learnable sharpness.

    The nine paths are candidates, not choices: their headings are blended in
    proportion to how safe the field says each one is, and the result is a
    direction the agent can travel in whether or not it is one of the nine.
    Any heading on the circle is reachable, because sliding weight from one
    neighbour to the next sweeps the sum continuously between them.

    Blending rather than picking is also what stops the twitching. An argmax
    flips the whole action the instant two paths swap order, however close the
    two readings were; a blend moves the heading by as much as the readings
    moved. The jitter was never the agent changing its mind -- it was a
    discrete output turning small changes of mind into large changes of course.

    A mean near zero is the field saying every direction is as bad as every
    other. The simulation reads that as standing still, which is what it means.
    """

    EPSILON = 1e-3

    def __init__(self, directions: torch.Tensor, temperature: float = INITIAL_TEMPERATURE) -> None:
        super().__init__()
        self.actions = int(directions.shape[0])
        # A buffer, so it travels into the checkpoint: which way each path
        # pointed is part of what the weights were trained against.
        self.register_buffer("directions", directions)
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(temperature))))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        safety = latent[:, : self.actions]
        # Standardise across the paths, so only the *shape* of the field
        # reaches the blend and its magnitude does not. Untouched, a freshly
        # initialised field spreads its readings far too little to move a
        # softmax, so every weight would sit at a ninth and the heading would
        # be the average of all nine -- zero -- until the field grew. It also
        # stops a field that drifts large from silently collapsing the blend
        # onto a single path, which is the argmax this exists to avoid.
        safety = safety - safety.mean(dim=1, keepdim=True)
        spread = safety.std(dim=1, keepdim=True).clamp_min(self.EPSILON)
        weights = torch.softmax(safety / spread * self.log_temperature.exp(), dim=1)
        return weights @ self.directions


class VelocityFlowRoyalePolicy(ActorCriticPolicy):
    """ActorCritic whose heading comes from the field, not a learned head."""

    def __init__(self, *args, architecture: str = ROYALE_ARCHITECTURE, **kwargs) -> None:
        # Accepted and kept here rather than forwarded: SB3 splats
        # `policy_kwargs` straight into this constructor, so a name recorded
        # there would otherwise reach `ActorCriticPolicy.__init__` as an
        # unexpected keyword. Swallowing it is what lets the name live in the
        # saved kwargs, which is the only part of a checkpoint that can say
        # what architecture wrote it.
        self.architecture = architecture
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # `log_std` is whatever SB3 built for the Gaussian and is left alone;
        # only the mean is taken over, which is the same trade the discrete
        # discrete version made with the logits.
        self.action_net = _FieldDirection(unit_directions(self.features_extractor.layout))
        # The optimizer was built over the head that was just replaced.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


def policy_kwargs_for(layout: Layout) -> dict[str, Any]:
    """What to hand PPO so the layout travels into the checkpoint.

    SB3 stores `policy_kwargs` in the saved zip, so the layout a policy was
    trained against is recoverable from the checkpoint alone -- which is what
    makes `require_loadable` possible without a running gym.

    ``pi=[]`` because the actor has no hidden layers: the heading is the
    field's own reading, and a layer between them would be a learned head,
    which is the thing this architecture exists not to have.
    """
    return {
        "architecture": ROYALE_ARCHITECTURE,
        "features_extractor_class": VelocityFlowRoyaleExtractor,
        "features_extractor_kwargs": {"layout": layout.as_dict()},
        "net_arch": {"pi": [], "vf": [VALUE_FEATURES]},
        "log_std_init": LOG_STD_INIT,
    }


@dataclass(frozen=True)
class Architecture:
    """Everything a run of one architecture needs, in one shape.

    A dataclass rather than a dict because the entries are not alike: the
    policy kwargs depend on the layout and the discounts do not. As a dict,
    one key would have been a callable sitting among plain values, and a
    caller that read them uniformly would have passed a function to PPO as
    `policy_kwargs`. Here the difference is a method, and the type says so.
    """

    name: str
    policy_class: type
    #: v2's values, converted from four-frame to one-frame decisions so the
    #: time scales are preserved: 0.99 ** (1/4) and 0.95 ** (1/4). They keep
    #: the horizons the same length in seconds; they do not make one-frame PPO
    #: equivalent to four-frame PPO. Starting values, to tune.
    gamma: float
    gae_lambda: float

    def policy_kwargs(self, layout: Layout) -> dict[str, Any]:
        return policy_kwargs_for(layout)

    def ppo_kwargs(self, layout: Layout) -> dict[str, Any]:
        """What to hand `PPO(...)` for a fresh model.

        The discounts are included here rather than left to the caller so that
        they cannot be forgotten, or hardcoded beside a table that already
        holds them.
        """
        return {
            "policy": self.policy_class,
            "policy_kwargs": self.policy_kwargs(layout),
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
        }

    def resume_kwargs(self) -> dict[str, Any]:
        """What to hand `PPO.load(...)` so a checkpoint takes these discounts.

        `load` otherwise restores whatever the checkpoint was saved with, so a
        run resumed after the table changed would silently keep the old
        values. `custom_objects` is how SB3 overrides a saved field.
        """
        return {"custom_objects": {"gamma": self.gamma, "gae_lambda": self.gae_lambda}}


ROYALE = Architecture(
    name=ROYALE_ARCHITECTURE,
    policy_class=VelocityFlowRoyalePolicy,
    gamma=0.99 ** (1 / 4),
    gae_lambda=0.95 ** (1 / 4),
)

#: Every architecture this trainer can build, by name.
ARCHITECTURES: dict[str, Architecture] = {ROYALE_ARCHITECTURE: ROYALE}


def _saved_data(path) -> dict[str, Any]:
    from stable_baselines3.common.save_util import load_from_zip_file

    data, _, _ = load_from_zip_file(path, load_data=True, device="cpu")
    return data or {}


def checkpoint_architecture(path) -> str:
    """Which architecture a saved checkpoint was trained with.

    A checkpoint with no architecture recorded is not a Royale checkpoint.
    Guessing would load somebody else's weights into this policy and report a
    successful resume.
    """
    data = _saved_data(path)
    name = (data.get("policy_kwargs") or {}).get("architecture")
    if name:
        if str(name) in SUPERSEDED_ARCHITECTURES:
            raise ProtocolError(
                f"{path} was trained with {name}: {SUPERSEDED_ARCHITECTURES[str(name)]}"
            )
        return str(name)
    extractor = (data.get("policy_kwargs") or {}).get("features_extractor_class")
    if extractor is VelocityFlowRoyaleExtractor:
        return ROYALE_ARCHITECTURE
    raise ProtocolError(
        f"{path} does not record a DodgeRoyale architecture; it is not a Royale checkpoint"
    )


def checkpoint_layout(path) -> Layout:
    """The observation layout a saved checkpoint was trained against."""
    data = _saved_data(path)
    kwargs = (data.get("policy_kwargs") or {}).get("features_extractor_kwargs") or {}
    raw = kwargs.get("layout")
    if raw is None:
        raise ProtocolError(
            f"{path} does not record an observation layout, so there is no way to tell "
            "whether this gym's observations are the ones it was trained on"
        )
    return Layout.from_json(raw if isinstance(raw, dict) else raw.as_dict())


def require_loadable(path, served: Layout) -> None:
    """Refuse a checkpoint this gym cannot feed, before a model is built.

    Both halves matter. A foreign checkpoint would load weights shaped for a
    different game. A Royale checkpoint trained against a different layout is
    worse: the shapes can match exactly while every channel means something
    else, so it would train on, and the run would look healthy.
    """
    architecture = checkpoint_architecture(path)
    if architecture != ROYALE_ARCHITECTURE:
        raise ProtocolError(
            f"{path} was trained as {architecture!r}; this trainer only builds "
            f"{ROYALE_ARCHITECTURE!r}"
        )
    checkpoint_layout(path).require_compatible(served)
