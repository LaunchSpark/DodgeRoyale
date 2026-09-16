"""The Royale policy, and the checkpoint compatibility rules around it.

Adapted from DodgeAI ``dodge/policies.py`` and ``dodge/velocity.py`` at
revision e32f222 (``_FieldLogits``, ``VelocityFlowPolicy``). DodgeAI is a
separate project; nothing here imports it, and a Royale checkpoint never needs
it to load.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from .protocol import Layout, ProtocolError
from .velocity import INITIAL_TEMPERATURE, VALUE_FEATURES, VelocityFlowRoyaleExtractor

__all__ = [
    "ARCHITECTURES",
    "ROYALE_ARCHITECTURE",
    "VelocityFlowRoyalePolicy",
    "checkpoint_architecture",
    "checkpoint_layout",
    "policy_kwargs_for",
    "require_loadable",
]

#: The only architecture this trainer builds. Named in checkpoints, so a run
#: can say what it was.
ROYALE_ARCHITECTURE = "velocity-flow-royale"


class _FieldLogits(nn.Module):
    """Turn path danger into action logits, with a learnable sharpness.

    The field's units are arbitrary -- nothing pins its scale -- so without a
    temperature the initial policy could be anywhere between uniform and
    one-hot, and entropy would be set by initialisation rather than by
    learning.
    """

    EPSILON = 1e-3

    def __init__(self, actions: int, temperature: float = INITIAL_TEMPERATURE) -> None:
        super().__init__()
        self.actions = actions
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(temperature))))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        danger = latent[:, : self.actions]
        # Standardise across the actions, so only the *shape* of the field
        # reaches the distribution and its magnitude does not. Untouched, a
        # freshly initialised field spreads its readings far too little to move
        # a softmax, so the policy would sit at uniform until the field grew,
        # learning nothing meanwhile. It also stops a field that drifts large
        # from silently sharpening the policy toward greedy.
        danger = danger - danger.mean(dim=1, keepdim=True)
        spread = danger.std(dim=1, keepdim=True).clamp_min(self.EPSILON)
        return danger / spread * self.log_temperature.exp()


class VelocityFlowRoyalePolicy(ActorCriticPolicy):
    """ActorCritic whose logits come from the field, not from a learned head."""

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        actions = len(self.features_extractor.layout.actions)
        self.action_net = _FieldLogits(actions)
        # The optimizer was built over the head that was just replaced.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


def policy_kwargs_for(layout: Layout) -> dict[str, Any]:
    """What to hand PPO so the layout travels into the checkpoint.

    SB3 stores `policy_kwargs` in the saved zip, so the layout a policy was
    trained against is recoverable from the checkpoint alone -- which is what
    makes `require_loadable` possible without a running gym.

    ``pi=[]`` because the actor has no hidden layers: the logits are the
    field's own reading, and a layer between them would be a learned head,
    which is the thing this architecture exists not to have.
    """
    return {
        "features_extractor_class": VelocityFlowRoyaleExtractor,
        "features_extractor_kwargs": {"layout": layout.as_dict()},
        "net_arch": {"pi": [], "vf": [VALUE_FEATURES]},
    }


#: Everything needed to build a run of each known architecture.
ARCHITECTURES: dict[str, dict[str, Any]] = {
    ROYALE_ARCHITECTURE: {
        "policy_class": VelocityFlowRoyalePolicy,
        "policy_kwargs": policy_kwargs_for,
        # v2's values, converted from four-frame to one-frame decisions so the
        # time scales are preserved: 0.99 ** (1/4) and 0.95 ** (1/4). They keep
        # the horizons the same length in seconds; they do not make one-frame
        # PPO equivalent to four-frame PPO. Starting values, to tune.
        "gamma": 0.99 ** (1 / 4),
        "gae_lambda": 0.95 ** (1 / 4),
    }
}


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
