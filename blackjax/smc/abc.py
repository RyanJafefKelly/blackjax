"""TODO: summary"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
from jax import vmap
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
    sim_x: ArrayTree  # raw simulated data per particle
    summaries: ArrayTree  # simulated summaries per particle


class SMCABCInfo(NamedTuple):
    """Information about the SMC-ABC sampler."""

    ancestors: Array  # indices selected by resampling
    acceptance_rate: float  # mean acceptance in the MCMC move
    num_moved: int
    R_next: int
    num_simulations: int
    B_SIM: int = 1  # number of simulations per MCMC move


class Distance:
    """ABC discrepancy."""

    def update(self, live_particles: ArrayTree) -> None:
        """
        Called *inside abc_step* with the summaries of current
        live particles (shape ``(N_alive, summary_dim)``).
        Default implementation does nothing.
        """
        return

    def __call__(self, summaries: ArrayTree) -> Array:
        """Return ρ for *one* simulated summary vector."""
        raise NotImplementedError


# class Euclidean(Distance):
#     def __init__(self, s_obs):
#         self.s_obs = s_obs

#     def __call__(self, s):
#         delta = jnp.nan_to_num(s - self.s_obs, nan=1e10)
#         return jnp.linalg.norm(delta)


# class AdaptiveMahalanobis(Distance):
#     def __init__(self, s_obs):
#         self.s_obs = s_obs
#         self.chol_inv = None  # Σ^{-½}

#     def update(self, live_summaries):  # <- only summaries
#         if live_summaries is None:
#             return
#         live = jnp.nan_to_num(live_summaries, nan=0.0)
#         centred = live - live.mean(0, keepdims=True)
#         cov = (centred.T @ centred) / (live.shape[0] - 1)
#         eps = 1e-3 * jnp.trace(cov) / cov.shape[0]  # ridge for stability
#         cov = cov + eps * jnp.eye(cov.shape[0])
#         self.chol_inv = jnp.linalg.inv(jnp.linalg.cholesky(cov))


#     def __call__(self, s):
#         delta = jnp.nan_to_num(s - self.s_obs, nan=0.0)
#         if self.chol_inv is None:
#             return jnp.linalg.norm(delta)  # first round
#         return jnp.linalg.norm(delta @ self.chol_inv.T)
# class Mahalanobis(Distance):
#     """Fixed–covariance Mahalanobis distance.

#     Parameters
#     ----------
#     s_obs
#         Observed summary vector (shape `(p,)`).
#     cov
#         Positive–definite `(p, p)` covariance matrix **of the summaries**.
#         The same matrix is used for every SMC iteration.
#     ridge
#         Small positive constant added to the diagonal of `cov`
#         before inverting, to avoid numerical failure when `cov`
#         is close to singular.
#     """

#     def __init__(self, s_obs: Array, cov: Array, ridge: float = 1e-6):
#         self.s_obs = s_obs

#         # add a ridge and pre‑compute Σ^{‑½}
#         p = cov.shape[0]
#         cov = cov + ridge * jnp.eye(p)
#         chol = jnp.linalg.cholesky(cov)
#         self.chol_inv = jnp.linalg.inv(chol)

#     # no update needed ---------------------------------------------------------
#     def update(self, *_):
#         return

#     # single‑vector distance ---------------------------------------------------
#     def __call__(self, s: Array) -> Array:
#         δ = jnp.nan_to_num(s - self.s_obs, nan=0.0)  # (p,)
#         z = δ @ self.chol_inv.T  # whiten
#         return jnp.linalg.norm(z)  # √(δᵀΣ⁻¹δ)


# class MMD:  # NOTE: unbiased MMD²
#     def __init__(self, y_obs: Array, gamma: float):
#         self.y_obs = y_obs  # shape (48,)
#         self.gamma = gamma
#         self._yy = 1.0  # k(y,y)=1 for Gaussian kernel

#     def _k(self, x, y):
#         return jnp.exp(-self.gamma * jnp.sum((x - y) ** 2))

#     def _pairwise_mean(self, X):
#         """Un‑biased ⟨k(xᵢ,xⱼ)⟩_{i≠j} for a (B,·) batch."""
#         B = X.shape[0]
#         # build Gram but mask diagonal
#         G = vmap(lambda xi: vmap(lambda xj: self._k(xi, xj))(X))(X)
#         mean_offdiag = (jnp.sum(G) - jnp.sum(jnp.diag(G))) / (B * (B - 1))
#         return mean_offdiag

#     def __call__(self, sim_batch: Array) -> Array:
#         k_xx = self._pairwise_mean(sim_batch)
#         k_xy = jnp.mean(vmap(lambda x: self._k(x, self.y_obs))(sim_batch))
#         mmd2 = k_xx + self._yy - 2.0 * k_xy
#         # jax.debug.print("MMD: k_xx={:.4f}, k_xy={:.4f}, MMD²={:.4f}", k_xx, k_xy, mmd2)
#         return jnp.sqrt(jnp.maximum(mmd2, 0.0))


class MMD:
    """
    Biased estimate of MMD² between
        P  = empirical on `sim_batch`      (size m ≥ 1)
        Q  = degenerate at `y_obs` (size n = 1)
    with a Gaussian kernel k_γ.

    For n = 1 the unbiased estimator is undefined, so we *must* use
    the biased version where all terms include their diagonals.
    """

    def __init__(self, y_obs: jnp.ndarray, gamma: float):
        self.y_obs = y_obs  # (d,)
        self.gamma = gamma

    # ------------------------------------------------------------------
    # scalar kernel   k(x,y) = exp(-γ‖x-y‖²)
    # ------------------------------------------------------------------
    def _k(self, x, y):
        return jnp.exp(-self.gamma * jnp.sum((x - y) ** 2))

    # ------------------------------------------------------------------
    # Mean of the Gram matrix *including* the diagonal
    #   (biased estimate; O(m²) but m = 10 here)
    # ------------------------------------------------------------------
    def _gram_mean(self, X):
        G = vmap(lambda xi: vmap(lambda xj: self._k(xi, xj))(X))(X)  # (m,m)
        return jnp.mean(G)  # scalar

    # ------------------------------------------------------------------
    # Callable: √MMD²  (biased)
    # ------------------------------------------------------------------
    def __call__(self, sim_batch: jnp.ndarray) -> jnp.ndarray:
        # m = sim_batch.shape[0]  # number of simulations
        k_xx = self._gram_mean(sim_batch)  # (1/m²)∑_{i,j} k(xi,xj)

        # k(yi,yj) with n = 1  →  k_yy = k(y,y) = 1
        k_yy = jnp.array(1.0, sim_batch.dtype)

        # (1/(mn)) ∑_{i,j} k(xi,yj)   but n = 1
        k_xy = jnp.mean(vmap(lambda x: self._k(x, self.y_obs))(sim_batch))

        mmd2 = k_xx + k_yy - 2.0 * k_xy
        return jnp.sqrt(jnp.maximum(mmd2, 0.0))


# helper function: default weight function for ABC
# def _make_weight_fn(
#     epsilon: float, distance_fn: Callable[[ArrayTree], Array]
# ) -> Callable[[ArrayTree], Array]:
#     """Return a `weight_fn` compatible with `smc_base.step`.

#     It assigns log‑weight 0 to particles with ρ≤ε and −∞ otherwise
#     (so they get discarded in the following *discard α* step).
#     """

#     def log_weights(particles: ArrayTree) -> Array:
#         dist = distance_fn(particles)
#         keep = dist <= epsilon
#         return jnp.where(keep, 0.0, -jnp.inf)

#     return log_weights


def init(
    key: PRNGKey,
    particles: ArrayLikeTree,
    epsilon: float,
    distance_fn: Callable[[ArrayTree], Array],
    simulate_fn: Callable[
        [PRNGKey, ArrayTree], ArrayTree
    ],  # returns raw x (batched if needed)
    summary_fn: Callable[[ArrayTree], ArrayTree],
    *,
    initial_R: int = 1,
) -> SMCABCState:
    """Create an initial `SMCABCState`."""
    first_leaf, *_ = jax.tree_util.tree_leaves(particles)
    N = first_leaf.shape[0]

    key, subkey = jax.random.split(key)
    sim_keys = jax.random.split(subkey, N)  # (N,)
    x0 = jax.vmap(simulate_fn)(sim_keys, particles)  # raw sims
    s0 = jax.vmap(summary_fn)(x0)  # summaries
    d0 = jax.vmap(distance_fn)(s0)  # distances
    w0 = jnp.ones(N) / N

    return SMCABCState(
        particles=particles,
        weights=jnp.squeeze(w0),
        epsilon=epsilon,
        distances=jnp.squeeze(d0),
        R=initial_R,
        cov_rw=None,
        sim_x=x0,
        summaries=s0,
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
    R_cur: int,
    *,
    simulate_fn: Callable[[PRNGKey, ArrayTree], ArrayTree],  # raw x
    summary_fn: Callable[[ArrayTree], ArrayTree],
    distance_fn: Callable[[ArrayTree], Array],
    prior_logpdf: Callable[[ArrayTree], Array],
    alpha: float = 0.5,
    c: float = 0.01,
    B_sim: int = 1,  # number of simulations per distance computation,
    resampling_fn: Callable = resampling.systematic,
) -> tuple[SMCABCState, SMCABCInfo]:
    """
    Replenishment SMC‑ABC step (Algorithms 1 & 2, Drovandi & Pettitt 2011).
    """
    N = state.weights.shape[0]
    Na = int(alpha * N)
    N_alive = N - Na
    # R_cur = state.R

    # discard α N worst by distance
    sort_idx = jnp.argsort(state.distances)
    keep_idx = sort_idx[:N_alive]
    alive_particles = state.particles[keep_idx]
    alive_x = state.sim_x[keep_idx]
    alive_s = state.summaries[keep_idx]
    alive_d = state.distances[keep_idx]
    epsilon_next = jnp.max(alive_d)
    # rng_key, live_key = jax.random.split(rng_key)
    # live_keys = jax.random.split(live_key, N_alive)
    # live_summaries = jax.vmap(simulate_fn)(live_keys, alive_particles)  # (N_alive,48)
    # alive_distances = jax.vmap(distance_fn)(live_summaries)
    # epsilon_next = float(jnp.max(alive_distances))

    # adaptive proposal covariance
    centred = alive_particles - alive_particles.mean(0, keepdims=True)
    cov_rw = (centred.T @ centred) / jnp.maximum(N_alive - 1, 1)
    # small ridge for stability
    ridge = 1e-8 * jnp.trace(cov_rw) / jnp.maximum(cov_rw.shape[0], 1)
    chol_S = jnp.linalg.cholesky(cov_rw + ridge * jnp.eye(cov_rw.shape[0]))

    # resample dropped slice
    rng_key, k_resample, k_mcmc = jax.random.split(rng_key, 3)
    resampled_idx = resampling_fn(k_resample, jnp.ones(N_alive) / N_alive, Na)
    th0 = alive_particles[resampled_idx]
    x0 = alive_x[resampled_idx]
    s0 = alive_s[resampled_idx]
    d0 = alive_d[resampled_idx]

    # R_cur Metropolis moves per resampled particle

    def mh_one(carry, key):
        th_old, x_old, s_old, d_old = carry
        k_prop, k_sim, k_u = jax.random.split(key, 3)
        step = jax.random.normal(k_prop, th_old.shape) @ chol_S.T
        th_prop = th_old + step
        x_prop = simulate_fn(k_sim, th_prop)
        s_prop = summary_fn(x_prop)
        d_prop = distance_fn(s_prop)
        log_alpha = jnp.where(
            d_prop <= epsilon_next,
            prior_logpdf(th_prop) - prior_logpdf(th_old),
            -jnp.inf,
        )
        accept = jnp.log(jax.random.uniform(k_u)) < jnp.minimum(0.0, log_alpha)

        th_new = jnp.where(accept, th_prop, th_old)
        x_new = jnp.where(accept, x_prop, x_old)
        s_new = jnp.where(accept, s_prop, s_old)
        d_new = jnp.where(accept, d_prop, d_old)
        return (th_new, x_new, s_new, d_new), accept

    def mh_chain(init_carry, keys_R):
        (th_f, x_f, s_f, d_f), accepts = jax.lax.scan(mh_one, init_carry, keys_R)
        return (th_f, x_f, s_f, d_f), accepts

    keys = jax.random.split(k_mcmc, Na * R_cur)
    keys_move = keys.reshape((Na, R_cur) + keys.shape[1:])
    (m_th, m_x, m_s, m_d), acc_matrix = jax.vmap(mh_chain)((th0, x0, s0, d0), keys_move)

    p_acc = jnp.mean(acc_matrix)
    R_next = jnp.maximum(
        1, jnp.ceil(jnp.log(c) / jnp.log(jnp.clip(1.0 - p_acc, 1e-12, 1.0)))
    ).astype(jnp.int32)
    num_moved = jnp.sum(jnp.any(acc_matrix, axis=1)).astype(jnp.int32)
    num_sim_step = int(Na * R_cur * B_sim)

    # assemble new population
    new_particles = jnp.concatenate([alive_particles, m_th], axis=0)
    new_x = jnp.concatenate([alive_x, m_x], axis=0)
    new_s = jnp.concatenate([alive_s, m_s], axis=0)
    new_d = jnp.concatenate([alive_d, m_d], axis=0)
    new_w = jnp.ones_like(state.weights) / N

    new_state = SMCABCState(
        particles=new_particles,
        weights=new_w,
        epsilon=epsilon_next,
        distances=new_d,
        R=R_next,
        cov_rw=cov_rw,
        sim_x=new_x,
        summaries=new_s,
    )

    info = SMCABCInfo(
        ancestors=resampled_idx,
        acceptance_rate=p_acc,
        num_moved=num_moved,
        R_next=R_next,
        num_simulations=num_sim_step,
        B_SIM=B_sim,
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
