"""Synthetic, paired through-ball counterfactuals, not an established evaluation.

Coordinates are metres: the defended goal is at x=0, the pitch extends to
x=105, and y=0 is the centre line. The ball moves at constant velocity. Both
players are point masses with instantaneous velocity changes and bounded speed;
there is no acceleration, ball height, offside, other players, or possession
after a touch. Player speed is capped at 12 m/s and ball speed at 50 m/s as
synthetic feasibility limits, not empirical measurements. The keeper may
intercept when within the contact radius of the
ball *and* the ball is inside the keeper's penalty area (including its edges).
This deliberately conservative proxy is not a full implementation of football
laws. Both players use the same contact radius; an attacker touch is legal
anywhere on the pitch. Simultaneous legal contacts are reported as contested.

For the first interval the keeper holds or moves along the x axis at the lesser
of its speed, 2 m/s, and 1 metre divided by the interval duration. Thus the
intervention is at most a one-metre controlled step. An action that would
cross the pitch boundary is rejected, not clipped.
Afterward every branch uses the same feedback rule: chase the ball. The attacker
chases the ball, optionally aiming away from the keeper. Explicit Euler control
is used, with linear motion between samples; the evaluator independently finds
continuous first contacts and goal/field crossings on those segments. Contacts
at a crossing take precedence over the crossing. A ball leaving the pitch
outside the goal, or no event by the horizon, is reported as unresolved.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional, Sequence, Tuple

import numpy as np


Point = Tuple[float, float]
ACTIONS = ("hold", "forward", "retreat")


@dataclass(frozen=True)
class Ball:
    position: Point
    velocity: Point


@dataclass(frozen=True)
class Player:
    position: Point
    speed: float


@dataclass(frozen=True)
class State:
    ball: Ball
    attacker: Player
    keeper: Player


@dataclass(frozen=True)
class Config:
    first_interval: float = 0.5
    horizon: float = 4.0
    dt: float = 0.02
    contact_radius: float = 0.6
    attacker_avoidance: float = 0.0
    ball_velocity_noise: float = 0.0
    attacker_speed_noise: float = 0.0


@dataclass(frozen=True)
class Draws:
    """Realized perturbations, shared verbatim across counterfactual branches."""

    ball_velocity: Point = (0.0, 0.0)
    attacker_speed: float = 0.0


@dataclass(frozen=True)
class Sample:
    time: float
    ball: Point
    attacker: Point
    keeper: Point


@dataclass(frozen=True)
class Outcome:
    kind: str
    time: float


@dataclass(frozen=True)
class Branch:
    action: str
    trajectory: Tuple[Sample, ...]
    outcome: Outcome


def _point(value, name):
    try:
        values = tuple(value)
        if not all(isinstance(v, Real) and not isinstance(v, bool) for v in values):
            raise ValueError
        result = tuple(float(v) for v in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite 2D point") from exc
    if len(result) != 2 or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must be a finite 2D point")
    return result


def _number(value, name, minimum=0.0, strict=False):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        raise ValueError(f"invalid {name}")
    return value


def _inside(point):
    return 0.0 <= point[0] <= 105.0 and -34.0 <= point[1] <= 34.0


def _validate(state, config, draws):
    for name, pos in (("ball", state.ball.position), ("attacker", state.attacker.position),
                      ("keeper", state.keeper.position)):
        if not _inside(_point(pos, name)):
            raise ValueError(f"{name} must start on the pitch")
    _point(state.ball.velocity, "ball velocity")
    ball_noise = _point(draws.ball_velocity, "ball velocity draw")
    if math.hypot(*state.ball.velocity) > 50 or math.hypot(
            *(a + b for a, b in zip(state.ball.velocity, ball_noise))) > 50 + 1e-12:
        raise ValueError("ball speed exceeds 50 m/s")
    for name, speed in (("attacker speed", state.attacker.speed),
                        ("keeper speed", state.keeper.speed)):
        if _number(speed, name) > 12:
            raise ValueError(f"{name} exceeds 12 m/s")
    speed_delta = _number_signed(draws.attacker_speed, "attacker speed draw")
    if not 0 <= state.attacker.speed + speed_delta <= 12:
        raise ValueError("realized attacker speed must be between 0 and 12 m/s")
    for name in ("first_interval", "horizon", "dt"):
        _number(getattr(config, name), name, strict=True)
    if config.first_interval > config.horizon:
        raise ValueError("first interval exceeds horizon")
    if config.dt < config.horizon / 199_999:
        raise ValueError("dt would exceed the 200000-step simulation limit")
    for name in ("contact_radius", "attacker_avoidance",
                 "ball_velocity_noise", "attacker_speed_noise"):
        _number(getattr(config, name), name)


def _number_signed(value, name):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def sample_draws(state: State, config: Config, seed: int) -> Draws:
    """Draw once per scene seed, with no action- or time-step-dependent RNG.

    Realized speeds are clipped to their physical limits; these latent draws
    are then fixed for every branch and every time-step resolution.
    """
    _validate(state, config, Draws())
    rng = np.random.default_rng(seed)
    ball = np.asarray(state.ball.velocity) + rng.normal(0.0, config.ball_velocity_noise, size=2)
    magnitude = np.linalg.norm(ball)
    if magnitude > 50:
        ball *= 50 / magnitude
    ball -= np.asarray(state.ball.velocity)
    speed = rng.normal(0.0, config.attacker_speed_noise)
    speed = min(12, max(0, state.attacker.speed + float(speed))) - state.attacker.speed
    return Draws((float(ball[0]), float(ball[1])), speed)


def _advance_toward(pos, target, speed, dt):
    target = np.clip(target, (0.0, -34.0), (105.0, 34.0))
    delta = target - pos
    distance = np.linalg.norm(delta)
    if distance == 0:
        return pos.copy()
    return pos + delta * min(1.0, speed * dt / distance)


def simulate(state: State, config: Config, action: str, draws: Draws = Draws()) -> Tuple[Sample, ...]:
    """Simulate one branch, rejecting actions that leave the pitch."""
    _validate(state, config, draws)
    if action not in ACTIONS:
        raise ValueError(f"unknown keeper action: {action!r}")
    direction = {"hold": 0.0, "forward": 1.0, "retreat": -1.0}[action]
    step_speed = min(state.keeper.speed, 2.0, 1.0 / config.first_interval)
    end_x = state.keeper.position[0] + direction * step_speed * config.first_interval
    if not 0.0 <= end_x <= 105.0:
        raise ValueError("first-interval action would leave the pitch")
    ball = np.asarray(_point(state.ball.position, "ball"), dtype=float)
    ball_velocity = np.asarray(_point(state.ball.velocity, "ball velocity")) + np.asarray(
        _point(draws.ball_velocity, "ball velocity draw"))
    attacker = np.asarray(_point(state.attacker.position, "attacker"), dtype=float)
    keeper = np.asarray(_point(state.keeper.position, "keeper"), dtype=float)
    attacker_speed = state.attacker.speed + draws.attacker_speed
    time = 0.0
    samples = [Sample(time, tuple(ball), tuple(attacker), tuple(keeper))]
    while time < config.horizon:
        if len(samples) > 200_000:
            raise ValueError("simulation exceeds the 200000-step limit")
        # Split exactly at the intervention boundary even if dt does not divide it.
        boundary = config.first_interval if time < config.first_interval else config.horizon
        step = min(config.dt, boundary - time, config.horizon - time)
        if step <= 0 or time + step == time:
            raise ValueError("dt is too small to advance simulation time")
        target = ball.copy()
        away = attacker - keeper
        distance = np.linalg.norm(away)
        if distance:
            target += config.attacker_avoidance * away / distance
        next_attacker = _advance_toward(attacker, target, attacker_speed, step)
        if time < config.first_interval:
            x = state.keeper.position[0] + direction * step_speed * min(time + step, config.first_interval)
            next_keeper = np.array((min(105.0, max(0.0, x)), keeper[1]))
        else:
            next_keeper = _advance_toward(keeper, ball, state.keeper.speed, step)
        ball = ball + ball_velocity * step
        attacker, keeper = next_attacker, next_keeper
        time = min(boundary, time + step)
        samples.append(Sample(time, tuple(ball), tuple(attacker), tuple(keeper)))
    return tuple(samples)


def _box_interval(start, end, lower, upper):
    """Portion of a segment inside a closed axis-aligned box, or None."""
    lo, hi = 0.0, 1.0
    for p, q, minimum, maximum in zip(start, end, lower, upper):
        delta = q - p
        if delta == 0:
            if not minimum <= p <= maximum:
                return None
        else:
            a, b = (minimum - p) / delta, (maximum - p) / delta
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
            if lo > hi:
                return None
    return lo, hi


def _contact_interval(ball0, ball1, player0, player1, radius):
    relative = np.asarray(ball0) - np.asarray(player0)
    delta = (np.asarray(ball1) - np.asarray(ball0)) - (np.asarray(player1) - np.asarray(player0))
    a = float(np.dot(delta, delta))
    b = 2.0 * float(np.dot(relative, delta))
    c = float(np.dot(relative, relative)) - radius * radius
    if a == 0:
        return (0.0, 1.0) if c <= 0 else None
    discriminant = b * b - 4.0 * a * c
    # Roundoff in the subtraction can make a tangent's discriminant negative.
    # Scale the tolerance to the terms being subtracted, not to world units.
    tolerance = 16.0 * np.finfo(float).eps * (b * b + abs(4.0 * a * c))
    if discriminant < -tolerance:
        return None
    root = math.sqrt(max(0.0, discriminant))
    lo, hi = max(0.0, (-b - root) / (2 * a)), min(1.0, (-b + root) / (2 * a))
    return (lo, hi) if lo <= hi else None


def evaluate(trajectory: Sequence[Sample], config: Config) -> Outcome:
    """Score sampled paths without using the action or simulator's control logic."""
    if not trajectory:
        raise ValueError("trajectory must not be empty")
    _number(config.contact_radius, "contact_radius")
    for i, sample in enumerate(trajectory):
        _number(sample.time, "sample time")
        for name in ("ball", "attacker", "keeper"):
            point = _point(getattr(sample, name), name)
            if name != "ball" and not _inside(point):
                raise ValueError(f"{name} outside pitch")
        if i and sample.time <= trajectory[i - 1].time:
            raise ValueError("sample times must increase")
    if not _inside(trajectory[0].ball):
        raise ValueError("ball must start on the pitch")
    # Include a zero-length segment so initial and single-sample contacts count.
    segments = [(trajectory[0], trajectory[0])] + list(zip(trajectory, trajectory[1:]))
    for start, end in segments:
        candidates = []
        pitch = _box_interval(start.ball, end.ball, (0.0, -34.0), (105.0, 34.0))
        penalty = _box_interval(start.ball, end.ball, (0.0, -20.16), (16.5, 20.16))
        keeper = _contact_interval(start.ball, end.ball, start.keeper, end.keeper, config.contact_radius)
        if keeper is not None and penalty is not None:
            lo, hi = max(keeper[0], penalty[0]), min(keeper[1], penalty[1])
            if lo <= hi:
                candidates.append((lo, 0, "keeper_interception"))
        attacker = _contact_interval(start.ball, end.ball, start.attacker, end.attacker, config.contact_radius)
        if attacker is not None and pitch is not None:
            lo, hi = max(attacker[0], pitch[0]), min(attacker[1], pitch[1])
            if lo <= hi:
                candidates.append((lo, 1, "attacker_touch"))
        dx = end.ball[0] - start.ball[0]
        dy = end.ball[1] - start.ball[1]
        reaches_boundary = ((end.ball[0] <= 0 and dx < 0) or
                            (end.ball[0] >= 105 and dx > 0) or
                            (end.ball[1] <= -34 and dy < 0) or
                            (end.ball[1] >= 34 and dy > 0))
        if pitch is not None and reaches_boundary:
            fraction = pitch[1]
            x = start.ball[0] + fraction * (end.ball[0] - start.ball[0])
            y = start.ball[1] + fraction * (end.ball[1] - start.ball[1])
            goal = dx < 0 and abs(x) < 1e-9 and abs(y) <= 3.66
            candidates.append((fraction, 2 if goal else 3, "goal" if goal else "unresolved"))
        if candidates:
            fraction, priority, kind = min(candidates)
            if priority < 2 and any(
                other_priority < 2 and other_priority != priority
                and abs(other_fraction - fraction) <= 1e-12
                for other_fraction, other_priority, _ in candidates
            ):
                kind = "contested"
            return Outcome(kind, start.time + fraction * (end.time - start.time))
    return Outcome("unresolved", trajectory[-1].time)


def compare_actions(state: State, config: Config, seed: int,
                    actions: Sequence[str] = ACTIONS) -> Tuple[Branch, ...]:
    """Return paired branches in requested order, with the same exogenous draws."""
    draws = sample_draws(state, config, seed)
    branches = []
    for action in actions:
        trajectory = simulate(state, config, action, draws)
        branches.append(Branch(action, trajectory, evaluate(trajectory, config)))
    return tuple(branches)
