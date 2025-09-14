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
    simulate_fn: Optional[Callable[[PRNGKey, ArrayTree], ArrayTree]],
    *,
    initial_R: int = 1,
) -> SMCABCState:
    """Create an initial `SMCABCState`."""
    first_leaf, *_ = jax.tree_util.tree_leaves(particles)
    N = first_leaf.shape[0]

    key, subkey = jax.random.split(key)
    sim_keys = jax.random.split(subkey, N)  # (N,)
    sims = jax.vmap(simulate_fn)(sim_keys, particles)  # (N,48)
    # sims = jnp.nan_to_num(sims, nan=0.0)
    distances = jax.vmap(distance_fn)(sims)
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
    R_cur: int,
    *,
    simulate_fn: Callable[[PRNGKey, ArrayTree], ArrayTree],
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

    alive_particles = jax.tree.map(lambda x: x[keep_idx], state.particles)
    alive_particles = alive_particles.reshape(alive_particles.shape[0], -1)
    epsilon_next = jnp.max(state.distances[keep_idx])  # next ε
    alive_distances = state.distances[keep_idx]
    # rng_key, live_key = jax.random.split(rng_key)
    # live_keys = jax.random.split(live_key, N_alive)
    # live_summaries = jax.vmap(simulate_fn)(live_keys, alive_particles)  # (N_alive,48)
    # alive_distances = jax.vmap(distance_fn)(live_summaries)
    # epsilon_next = float(jnp.max(alive_distances))

    # adaptive proposal covariance
    centred = alive_particles - alive_particles.mean(0)
    cov_rw = (centred.T @ centred) / (N_alive - 1)

    # add ridge for stability
    # var_diag = jnp.var(alive_particles, axis=0)
    # eps_ridge = 1e-6 * jnp.maximum(var_diag, 1.0)
    # cov_rw = cov_rw + jnp.diag(eps_ridge)  # (d,d)

    # d = cov_rw.shape[0]
    # TODO?  optimal Random‑Walk MH in d dims uses Σ_rw = (2.38²/d) · Σ_emp.
    # cov_rw *= (2.38**2) / d
    # chol_S = jnp.linalg.cholesky(cov_rw + 1e-8 * jnp.eye(d))
    chol_S = jnp.linalg.cholesky(jnp.atleast_2d(cov_rw))
    # chol_S = jnp.linalg.cholesky(cov_rw + 1e-6 * jnp.eye(cov_rw.shape[0]))

    # resample dropped slice
    rng_key, k_resample, k_mcmc = jax.random.split(rng_key, 3)
    resampled_idx = resampling_fn(k_resample, jnp.ones(N_alive) / N_alive, Na)
    proposal_particles = alive_particles[resampled_idx]  # (Na, d)

    # R_cur Metropolis moves per resampled particle

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
    keys = jax.random.split(k_mcmc, Na * R_cur)
    keys_move = keys.reshape((Na, R_cur) + keys.shape[1:])
    moved_particles, acc_matrix = jax.vmap(mh_chain)(proposal_particles, keys_move)
    # moved_particles (Na, d), acc_matrix (Na, R_cur)

    p_acc = jnp.mean(acc_matrix)
    R_next = jnp.maximum(
        1, jnp.ceil(jnp.log(c) / jnp.log(jnp.clip(1.0 - p_acc, 1e-12, 1.0)))
    ).astype(jnp.int32)
    num_moved = jnp.sum(jnp.any(acc_matrix, axis=1)).astype(jnp.int32)

    num_sim_step = int(Na * (R_cur + 1) * B_sim)

    # recompute distances for moved set
    rng_sim_keys = jax.random.split(rng_key, Na)
    moved_distances = jax.vmap(lambda k, th: distance_fn(simulate_fn(k, th)))(
        rng_sim_keys, moved_particles
    )

    # moved_distances = jnp.squeeze(moved_distances)

    # assemble new population
    new_particles = jnp.concatenate([alive_particles, moved_particles], axis=0)
    new_distances = jnp.concatenate([alive_distances, moved_distances], axis=0)
    new_weights = jnp.ones_like(state.weights) / N

    new_state = SMCABCState(
        particles=new_particles,
        weights=new_weights,
        epsilon=epsilon_next,
        distances=new_distances,
        R=R_next,
        cov_rw=cov_rw,
    )

    info = SMCABCInfo(
        ancestors=resampled_idx,
        acceptance_rate=p_acc,
        num_moved=num_moved,
        R_next=R_next,
        num_simulations=num_sim_step,
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
