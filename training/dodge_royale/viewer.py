"""Watch the agent: what the policy sees, and what it decides to do about it.

Royale cannot show what DodgeAI's watcher showed. That one ran a Python
reimplementation of the game and drew the arena; here the simulation is Rust
and protocol v1 carries the observation, not the world. So this draws the
observation instead -- the 256-pixel window the policy actually reads, the
paths it is choosing between, and the danger it assigns each one.

That turns out to be the more useful picture. Watching the arena shows what
happened; watching the observation shows *why*, because it is exactly the
information the policy had. A dodge into a threat is a bug in the field; a
dodge into a wall of nothing is a bug in the paths.

Snapshots are picked up between episodes, never during one. A policy that
changed mid-episode would make the episode unattributable to any update, and a
snapshot that will not load leaves the running policy alone so the next
episode tries again rather than the viewer dying with the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .protocol import Layout
from .rewards import Rewards
from .snapshots import newest_snapshot
from .vec_env import RoyaleVecEnv

__all__ = ["Frame", "WatchSession", "render_observation"]

#: Colour per hazard channel, in the order a cell's owner is resolved. Blast
#: outranks kamikaze outranks normal, matching the encoder's own priority, so
#: a contested cell is drawn as whatever actually owns it.
CHANNEL_COLOURS: dict[str, tuple[int, int, int]] = {
    "normal-enemy": (220, 70, 70),
    "kamikaze": (240, 150, 40),
    "blast": (250, 230, 90),
    "player": (255, 255, 255),
}

BACKGROUND = (18, 24, 38)
GRID_LINE = (30, 40, 60)
PATH_COLOUR = (90, 200, 255)
CHOSEN_COLOUR = (120, 255, 160)


def render_observation(
    observation: np.ndarray,
    layout: Layout,
    *,
    scale: int = 8,
    chosen: int | None = None,
    show_paths: bool = True,
) -> np.ndarray:
    """Draw one observation as an RGB image.

    Returns `(grid * scale, grid * scale, 3)` of uint8. Nearest-neighbour
    upscaling on purpose: a cell is a unit of information here, and smoothing
    between cells would draw danger where the policy was told there was none.
    """
    grid = layout.grid
    cells = grid * grid
    start = layout.grid_section.offset
    channels = {
        name: observation[start + index * cells : start + (index + 1) * cells].reshape(
            grid, grid
        )
        for index, name in enumerate(layout.channels)
    }

    image = np.zeros((grid, grid, 3), dtype=np.uint8)
    image[:, :] = BACKGROUND
    # Faint cell lines every eight cells, so distance is readable.
    image[::8, :] = GRID_LINE
    image[:, ::8] = GRID_LINE

    # Painted lowest priority first, so the owner of a contested cell wins,
    # matching how the encoder resolved it.
    for name in ("normal-enemy", "kamikaze", "blast", "player"):
        plane = channels.get(name)
        if plane is None:
            continue
        mask = plane > 0
        if name == "blast" and "blast-phase" in channels:
            # Phase is +1 just detonated, 0 at its widest, -1 about to
            # vanish. Only the expiring half fades: at phase 0 a blast is at
            # its largest and is exactly as lethal as a fresh one, so dimming
            # it there would draw the most dangerous moment as the safest.
            phase = channels["blast-phase"]
            weight = np.clip(1.0 + phase, 0.3, 1.0)[..., None]
            colour = np.array(CHANNEL_COLOURS[name], dtype=np.float32) * weight
            image[mask] = colour[mask].astype(np.uint8)
        else:
            image[mask] = CHANNEL_COLOURS[name]

    large = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

    if show_paths:
        paths = observation[
            layout.path_section.offset : layout.path_section.stop
        ].reshape(len(layout.actions), len(layout.horizons), 2)
        centre = grid * scale / 2.0
        for action in range(paths.shape[0]):
            colour = CHOSEN_COLOUR if action == chosen else PATH_COLOUR
            radius = 2 if action == chosen else 1
            for horizon in range(paths.shape[1]):
                dx, dy = paths[action, horizon]
                x = int(centre + dx * centre)
                y = int(centre + dy * centre)
                if 0 <= x < large.shape[1] and 0 <= y < large.shape[0]:
                    lo_y, hi_y = max(y - radius, 0), min(y + radius + 1, large.shape[0])
                    lo_x, hi_x = max(x - radius, 0), min(x + radius + 1, large.shape[1])
                    large[lo_y:hi_y, lo_x:hi_x] = colour
    return large


@dataclass
class Frame:
    """One step of a watched episode."""

    image: np.ndarray
    action: int
    action_name: str
    frame: int
    episode: int
    update: int | None
    danger: tuple[float, ...] = ()
    done: bool = False
    died: bool = False

    @property
    def seconds(self) -> float:
        return self.frame / 60.0


class WatchSession:
    """A one-env gym driven by the newest published policy.

    Its own gym, deliberately: watching must not perturb the run being
    watched, and a single env at a modest enemy count costs little.
    """

    def __init__(
        self,
        snapshots: str | Path,
        *,
        enemies: int = 100,
        max_frames: int = 3600,
        seed: int = 0,
        deterministic: bool = True,
        binary: str | None = None,
    ) -> None:
        self.snapshots = Path(snapshots)
        self.deterministic = deterministic
        self.env = RoyaleVecEnv(
            envs=1,
            enemies=enemies,
            max_frames=max_frames,
            threads=1,
            seed=seed,
            binary=binary,
            rewards=Rewards(),
        )
        self.model = None
        self.update: int | None = None
        self.episode = 0
        self.frame = 0
        self.error: str | None = None
        self._observation = self.env.reset()
        self.reload()

    # -- the policy --

    def reload(self) -> bool:
        """Adopt the newest snapshot above the one already loaded.

        A snapshot that will not load leaves the running policy untouched and
        the update number unchanged, so the next episode tries again rather
        than the viewer dying with the file.
        """
        found = newest_snapshot(self.snapshots, after=self.update if self.update else -1)
        if found is None:
            return False
        path, update = found
        try:
            from stable_baselines3 import PPO

            self.model = PPO.load(path, device="cpu")
            self.update = update
            self.error = None
            return True
        except Exception as error:  # noqa: BLE001 - reported, not fatal
            self.error = f"could not load {path.name}: {error}"
            return False

    def _act(self) -> tuple[int, tuple[float, ...]]:
        if self.model is None:
            # No policy published yet: stand still rather than act at random,
            # so what is on screen is never mistaken for a decision.
            return 0, ()
        actions, _ = self.model.predict(
            self._observation, deterministic=self.deterministic
        )
        danger: tuple[float, ...] = ()
        try:
            import torch

            with torch.no_grad():
                features = self.model.policy.extract_features(
                    torch.as_tensor(self._observation).to(self.model.device)
                )
            # The leading values are the negated danger the controller read
            # along each path, which is the reasoning behind the choice.
            danger = tuple(
                float(value) for value in features[0, : len(self.env.layout.actions)]
            )
        except Exception:  # noqa: BLE001 - a readout, not the point
            danger = ()
        return int(np.asarray(actions).reshape(-1)[0]), danger

    def step(self, *, scale: int = 8) -> Frame:
        """Advance one frame and draw it."""
        action, danger = self._act()
        observation, _, dones, infos = self.env.step(np.array([action]))
        self.frame += 1
        done = bool(dones[0])
        died = bool(infos[0].get("episode_summary", {}).get("died", False)) if done else False

        image = render_observation(
            self._observation[0], self.env.layout, scale=scale, chosen=action
        )
        frame = Frame(
            image=image,
            action=action,
            action_name=self.env.layout.actions[action],
            frame=self.frame,
            episode=self.episode,
            update=self.update,
            danger=danger,
            done=done,
            died=died,
        )
        self._observation = observation
        if done:
            # Between episodes, never during one: a policy that changed
            # mid-episode would make the episode attributable to no update.
            self.episode += 1
            self.frame = 0
            self.reload()
        return frame

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "WatchSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
