"""TODO: summary"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.smc import base as smc_base
from blackjax.smc import resampling  # reuse existing schemes


class SMCABCState(NamedTuple):
    """Current state of the SMC-ABC sampler."""

    particles: ArrayTree
    weights: Array  # shape (N,)
    epsilon: float
    distances: Array  # shape (N,)
    num_mcmc_moves: int


class SMCABCInfo(NamedTuple):
    """Information about the SMC-ABC sampler."""

    ancestors: Array  # indices selected by resampling
    acceptance_rate: float  # mean acceptance in the MCMC move
    num_moved: int  # how many resampled particles were moved ≥ 1 time
    epsilon_next: float  # εₜ₊₁ proposed for the next iteration


# helper function: default weight function for ABC
def _make_weight_fn(
    epsilon: float, distance_fn: Callable[[ArrayTree], Array]
) -> Callable[[ArrayTree], Array]:
    """Return a `weight_fn` compatible with `smc_base.step`.

    It assigns log‑weight 0 to particles with ρ≤ε and −∞ otherwise
    (so they get discarded in the following *discard α* step).
    """

    def log_weights(particles: ArrayTree) -> Array:
        dist = distance_fn(particles)
        keep = dist <= epsilon
        return jnp.where(keep, 0.0, -jnp.inf)

    return log_weights


def init(
    particles: ArrayLikeTree, epsilon: float, distance_fn: Callable[[ArrayTree], Array]
) -> SMCABCState:
    """Create an initial `SMCABCState`."""
    flat_particles = jax.tree_util.tree_leaves(particles)[0]
    num_particles = flat_particles.shape[0]

    distances = distance_fn(particles)
    weights = jnp.ones(num_particles) / num_particles
    return SMCABCState(particles, weights, epsilon, distances, num_mcmc_moves=0)


def step() -> tuple[SMCABCState, SMCABCInfo]:
    # TODO: implement the SMC ABC step
    return None, None


# TODO: see if this belongs in a different file
def smc_abc(
    *,
    simulate_fn: Callable[[PRNGKey, ArrayTree], ArrayTree],
    summary_fn: Callable[[ArrayTree], ArrayTree] = lambda x: x,
    distance: str | Callable[[ArrayTree], Array] = "euclidean",
    alpha: float = 0.5,
    resampling_fn: Callable = resampling.systematic,
    **mcmc_kwargs,
):
    """
    Return a `blackjax.base.SamplingAlgorithm` instance whose ``init/step`` follow the
    ABC‑SMC replenishment scheme.

    Parameters mirror those of :func:`step`; they are **closed over** in the returned object.
    """

    # Map string → actual distance function ---------------------------------
    if isinstance(distance, str):
        if distance.lower() == "euclidean":
            distance_fn = lambda sims: jnp.linalg.norm(
                summary_fn(sims) - obs_summary, axis=-1
            )
        else:
            raise ValueError(f"Unknown built‑in distance '{distance}'")
    else:
        distance_fn = distance

    def _init_fn(initial_particles: ArrayLikeTree, epsilon: float, *, rng_key=None):
        del rng_key
        return init(initial_particles, epsilon, distance_fn)

    def _step_fn(rng_key: PRNGKey, state: SMCABCState):
        return step(
            rng_key,
            state,
            simulate_fn=simulate_fn,
            distance_fn=distance_fn,
            alpha=alpha,
            resampling_fn=resampling_fn,
            **mcmc_kwargs,
        )

    from blackjax.base import SamplingAlgorithm

    return SamplingAlgorithm(_init_fn, _step_fn)
