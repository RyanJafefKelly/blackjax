"""TODO: summary"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array

from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.smc import resampling  # reuse existing schemes


class SMCABCState(NamedTuple):
    """Current state of the SMC-ABC sampler."""

    particles: ArrayTree
    weights: Array  # shape (N,)
    epsilon: float
    distances: Array  # shape (N,)
    R: int  # ← value to use in the next iteration
    cov_rw: Array | None  # proposal Σ (None at t=0)


class SMCABCInfo(NamedTuple):
    """Information about the SMC-ABC sampler."""

    ancestors: Array  # indices selected by resampling
    acceptance_rate: float  # mean acceptance in the MCMC move
    num_moved: int
    R_next: int


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
    key: PRNGKey,
    particles: ArrayLikeTree,
    epsilon: float,
    distance_fn: Callable[[ArrayTree], Array],
    simulate_fn: Optional[Callable[[PRNGKey, ArrayTree], ArrayTree]],
    *,
    initial_R: int = 1,
) -> SMCABCState:
    """Create an initial `SMCABCState`."""
    # TODO: eps_1 tolerance ?
    first_leaf, *_ = jax.tree_util.tree_leaves(particles)
    N = first_leaf.shape[0]

    key, subkey = jax.random.split(key)
    sim_keys = jax.random.split(subkey, N)  # (N,)
    sims = jax.vmap(simulate_fn)(sim_keys, particles)  # (N, 2)
    distances = jax.vmap(distance_fn)(sims)  # (N,)
    weights = jnp.ones(N) / N  # uniform at t = 0

    distances = jnp.squeeze(distances)  # shape (N,)
    weights = jnp.squeeze(weights)  # shape (N,)

    return SMCABCState(
        particles=particles,
        weights=weights,
        epsilon=epsilon,
        distances=distances,
        R=initial_R,
        cov_rw=None,  # will be set after first discard step
    )


def _batched_distance(particles, simulate_fn, distance_fn):
    """Return distances ρ_i for a batch of θ_i."""
    keys = jax.random.split(
        jax.random.PRNGKey(0), particles.shape[0]
    )  # replace by caller
    sim_summaries = jax.vmap(simulate_fn)(keys, particles)
    return jax.vmap(distance_fn)(sim_summaries)


def _rw_proposal(rng_key, theta, chol_S):
    # TODO: Note: could make more general / could be user-defined
    step = jax.random.normal(rng_key, theta.shape) @ chol_S.T
    return theta + step


def default_stopping(state: SMCABCState, info: SMCABCInfo, eps_min=1e-3, acc_min=0.10):
    return (state.epsilon <= eps_min) or (info.acceptance_rate < acc_min)


def abc_step(
    rng_key: PRNGKey,
    state: SMCABCState,
    *,
    simulate_fn: Callable[[PRNGKey, ArrayTree], ArrayTree],
    distance_fn: Callable[[ArrayTree], float],
    prior_logpdf: Callable[[ArrayTree], float],
    alpha: float = 0.5,
    c: float = 0.01,
    resampling_fn: Callable = resampling.systematic,
) -> tuple[SMCABCState, SMCABCInfo]:
    """
    Replenishment SMC‑ABC step (Algorithms 1 & 2, Drovandi & Pettitt 2011).
    """
    # --- unpack ----------------------------------------------------------------
    N = state.weights.shape[0]
    Na = int(jnp.floor(alpha * N))
    N_alive = N - Na
    R_cur = state.R

    # --- 1. discard α N worst by distance --------------------------------------
    sort_idx = jnp.argsort(state.distances)
    keep_idx = sort_idx[:N_alive]

    alive_particles = state.particles[keep_idx]
    alive_distances = state.distances[keep_idx]
    epsilon_next = alive_distances.max()

    # --- 2. adaptive proposal covariance --------------------------------------
    centred = alive_particles - alive_particles.mean(0)
    cov_rw = (centred.T @ centred) / (N_alive - 1)
    chol_S = jnp.linalg.cholesky(cov_rw + 1e-6 * jnp.eye(cov_rw.shape[0]))

    # --- 3. resample dropped slice --------------------------------------------
    rng_key, k_resample, k_mcmc = jax.random.split(rng_key, 3)
    resampled_idx = resampling_fn(k_resample, jnp.ones(N_alive) / N_alive, Na)
    proposal_particles = alive_particles[resampled_idx]  # (Na, d)

    # --- 4. R_cur Metropolis moves per resampled particle ---------------------
    keys_move = jax.random.split(k_mcmc, Na * R_cur).reshape(
        Na, R_cur, 2
    )  # (Na, R_cur, 2)

    def mh_one(key, theta_old):
        key_prop, key_sim, key_u = jax.random.split(key, 3)
        theta_prop = _rw_proposal(key_prop, theta_old, chol_S)
        rho_prop = distance_fn(simulate_fn(key_sim, theta_prop))

        # NOTE: proposal not included as symmetric, if make general later need to add this in
        log_alpha = jnp.where(
            rho_prop <= epsilon_next,
            prior_logpdf(theta_prop) - prior_logpdf(theta_old),
            -jnp.inf,
        )
        u = jax.random.uniform(key_u)
        accept = jnp.log(u) < jnp.minimum(0.0, log_alpha)

        theta_new = jnp.where(accept, theta_prop, theta_old)
        return theta_new, accept

    def mh_chain(theta0, keys_R):
        theta, accepts = jax.lax.scan(lambda th, k: mh_one(k, th), theta0, keys_R)
        return theta, accepts  # accepts shape (R_cur,)

    # vmap over Na (particles) so both inputs lead with axis 0 = Na
    moved_particles, acc_matrix = jax.vmap(mh_chain)(
        proposal_particles, keys_move
    )  # moved_particles (Na,d)

    p_acc = jnp.mean(acc_matrix)  # Alg. 2.15
    R_next = jnp.maximum(
        1, jnp.ceil(jnp.log(c) / jnp.log(jnp.clip(1.0 - p_acc, 1e-12, 1.0)))
    ).astype(
        int
    )  # Alg. 2.16
    num_moved = jnp.sum(jnp.any(acc_matrix, axis=1))

    # --- 5. recompute distances for moved set ----------------------------------
    rng_sim_keys = jax.random.split(rng_key, Na)
    moved_distances = jax.vmap(lambda k, th: distance_fn(simulate_fn(k, th)))(
        rng_sim_keys, moved_particles
    )
    moved_distances = jnp.squeeze(moved_distances)

    # --- 6. assemble new population -------------------------------------------
    new_particles = jnp.concatenate([alive_particles, moved_particles], axis=0)
    new_distances = jnp.concatenate([alive_distances, moved_distances], axis=0)
    new_weights = jnp.ones_like(state.weights) / N

    new_state = SMCABCState(
        particles=new_particles,
        weights=new_weights,
        epsilon=float(epsilon_next),
        distances=new_distances,
        R=int(R_next),
        cov_rw=cov_rw,
    )

    info = SMCABCInfo(
        ancestors=resampled_idx,
        acceptance_rate=float(p_acc),
        num_moved=int(num_moved),
        R_next=int(R_next),
    )
    return new_state, info


# # TODO: see if this belongs in a different file
# def smc_abc(
#     *,
#     simulate_fn: Callable[[PRNGKey, ArrayTree], ArrayTree],
#     summary_fn: Callable[[ArrayTree], ArrayTree] = lambda x: x,
#     distance: str | Callable[[ArrayTree], Array] = "euclidean",
#     alpha: float = 0.5,
#     resampling_fn: Callable = resampling.systematic,
#     **mcmc_kwargs,
# ):
#     """
#     Return a `blackjax.base.SamplingAlgorithm` instance whose ``init/step`` follow the
#     ABC‑SMC replenishment scheme.

#     Parameters mirror those of :func:`step`; they are **closed over** in the returned object.
#     """

#     # Map string → actual distance function ---------------------------------
#     if isinstance(distance, str):
#         if distance.lower() == "euclidean":
#             distance_fn = lambda sims: jnp.linalg.norm(
#                 summary_fn(sims) - obs_summary, axis=-1
#             )
#         else:
#             raise ValueError(f"Unknown built‑in distance '{distance}'")
#     else:
#         distance_fn = distance

#     def _init_fn(initial_particles: ArrayLikeTree, epsilon: float, *, rng_key=None):
#         del rng_key
#         return init(initial_particles, epsilon, distance_fn)

#     def _step_fn(rng_key: PRNGKey, state: SMCABCState):
#         return abc_step(
#             rng_key,
#             state,
#             simulate_fn=simulate_fn,
#             distance_fn=distance_fn,
#             alpha=alpha,
#             resampling_fn=resampling_fn,
#             **mcmc_kwargs,
#         )

#     from blackjax.base import SamplingAlgorithm

#     return SamplingAlgorithm(_init_fn, _step_fn)
