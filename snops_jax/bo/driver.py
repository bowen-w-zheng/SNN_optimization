"""
Main Bayesian Optimization driver loop for SNOPS.

Implements the complete BO algorithm with feasibility, intensification, and constrained EI.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable, Dict
import chex
from pathlib import Path
import json
from datetime import datetime

from snops_jax.bo.gp import fit_gp, fit_feasibility_gp
from snops_jax.bo.acquisition import compute_constrained_ei
from snops_jax.bo.suggest import suggest_candidates, initialize_bo
from snops_jax.bo.intensify import (
    intensification_loop,
    run_feasibility_check,
    IntensificationConfig,
    FeasibilityThresholds,
)


class BOConfig(NamedTuple):
    """Configuration for Bayesian Optimization."""

    # Initialization
    n_init: int = 50  # Number of LHS initial samples (paper default)

    # GP settings
    n_gp_restarts: int = 10  # Random restarts for GP hyperparameter optimization

    # Acquisition
    n_acq_restarts: int = 50  # Random restarts for acquisition optimization
    batch_size: int = 1  # Number of candidates per iteration

    # Intensification
    intensify_config: IntensificationConfig = IntensificationConfig()

    # Feasibility
    feasibility_thresholds: FeasibilityThresholds = FeasibilityThresholds()
    use_feasibility_gp: bool = True

    # Termination
    max_iterations: int = 200
    cost_threshold: float = 0.01  # Stop if cost below this

    # Logging
    log_dir: str = "./snops_logs"
    save_every: int = 10  # Save checkpoint every N iterations


class BOState(NamedTuple):
    """State of Bayesian Optimization."""

    # Evaluated parameters and results
    X_train: chex.Array  # (n_eval, n_dims) all evaluated parameters
    y_train: chex.Array  # (n_eval,) costs
    feasible: chex.Array  # (n_eval,) binary feasibility flags
    n_reps: chex.Array  # (n_eval,) number of repetitions per point

    # Current incumbent
    incumbent_idx: int
    incumbent_cost: float
    incumbent_sd: float

    # Iteration counter
    iteration: int


def run_bo_loop(
    objective_fn: Callable,
    bounds: chex.Array,  # (n_dims, 2)
    config: BOConfig,
    rng_key: chex.PRNGKey,
    initial_X: chex.Array = None,
) -> BOState:
    """
    Run Bayesian Optimization loop.

    Args:
        objective_fn: Function(theta, rep) -> (cost, stats, is_feasible)
        bounds: Parameter bounds
        config: BO configuration
        rng_key: Random key
        initial_X: Optional initial parameter sets (n_init, n_dims)

    Returns:
        Final BO state
    """
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    if initial_X is None:
        key_init, rng_key = jax.random.split(rng_key)
        initial_X = initialize_bo(bounds, config.n_init, key_init)

    n_init = initial_X.shape[0]
    n_dims = bounds.shape[0]

    # Evaluate initial points
    print(f"Evaluating {n_init} initial points...")
    X_train = []
    y_train = []
    feasible = []
    n_reps_list = []

    for i, theta in enumerate(initial_X):
        print(f"  Initial point {i+1}/{n_init}")

        # Check feasibility first (short run)
        cost_i, stats_i, is_feas = objective_fn(theta, rep=0)

        X_train.append(theta)
        y_train.append(cost_i)
        feasible.append(1.0 if is_feas else 0.0)
        n_reps_list.append(1)

    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train)
    feasible = jnp.array(feasible)
    n_reps = jnp.array(n_reps_list)

    # Find initial incumbent (best feasible point)
    feasible_mask = feasible > 0.5
    if jnp.sum(feasible_mask) > 0:
        feasible_costs = jnp.where(feasible_mask, y_train, jnp.inf)
        incumbent_idx = int(jnp.argmin(feasible_costs))
    else:
        # No feasible points yet
        incumbent_idx = int(jnp.argmin(y_train))

    incumbent_cost = float(y_train[incumbent_idx])
    incumbent_sd = 0.0

    state = BOState(
        X_train=X_train,
        y_train=y_train,
        feasible=feasible,
        n_reps=n_reps,
        incumbent_idx=incumbent_idx,
        incumbent_cost=incumbent_cost,
        incumbent_sd=incumbent_sd,
        iteration=0,
    )

    # Main BO loop
    for iteration in range(config.max_iterations):
        print(f"\n=== Iteration {iteration+1}/{config.max_iterations} ===")
        print(f"Incumbent cost: {state.incumbent_cost:.4f}")

        # Check termination
        if state.incumbent_cost < config.cost_threshold:
            print(f"Cost threshold reached: {state.incumbent_cost:.4f} < {config.cost_threshold}")
            break

        # Fit GPs
        key_gp, rng_key = jax.random.split(rng_key)
        print("Fitting cost GP...")
        cost_gp = fit_gp(state.X_train, state.y_train, config.n_gp_restarts, key_gp)

        if config.use_feasibility_gp and jnp.sum(state.feasible > 0.5) >= 5:
            print("Fitting feasibility GP...")
            key_feas, rng_key = jax.random.split(rng_key)
            feasibility_gp = fit_feasibility_gp(
                state.X_train, state.feasible, config.n_gp_restarts, key_feas
            )
        else:
            feasibility_gp = None

        # Suggest new candidates
        key_suggest, rng_key = jax.random.split(rng_key)
        print("Suggesting candidates...")
        candidates = suggest_candidates(
            cost_gp,
            feasibility_gp,
            bounds,
            state.incumbent_cost,
            config.batch_size,
            config.n_acq_restarts,
            use_feasibility=(feasibility_gp is not None),
            rng_key=key_suggest,
        )

        # Evaluate candidates with intensification
        new_X = []
        new_y = []
        new_feas = []
        new_n_reps = []

        for cand_idx, theta in enumerate(candidates):
            print(f"\nCandidate {cand_idx+1}/{config.batch_size}")

            # First repetition (with feasibility check)
            cost_0, stats_0, is_feas = objective_fn(theta, rep=0)

            if not is_feas:
                print(f"  Infeasible")
                new_X.append(theta)
                new_y.append(cost_0)
                new_feas.append(0.0)
                new_n_reps.append(1)
                continue

            print(f"  Feasible, cost = {cost_0:.4f}")

            # Intensification (if potentially optimal)
            costs_rep = [cost_0]

            from snops_jax.bo.intensify import is_potentially_optimal, should_stop_early

            if is_potentially_optimal(
                cost_0, state.incumbent_cost, state.incumbent_sd, config.intensify_config
            ):
                print(f"  Potentially optimal, intensifying...")

                for rep in range(1, config.intensify_config.max_reps):
                    cost_rep, _, _ = objective_fn(theta, rep=rep)
                    costs_rep.append(cost_rep)
                    print(f"    Rep {rep+1}: cost = {cost_rep:.4f}")

                    if should_stop_early(jnp.array(costs_rep), config.intensify_config):
                        print(f"    Early stop (SD < {config.intensify_config.sd_threshold})")
                        break

            mean_cost = float(jnp.mean(jnp.array(costs_rep)))
            sd_cost = float(jnp.std(jnp.array(costs_rep)))

            print(f"  Mean cost: {mean_cost:.4f} ± {sd_cost:.4f}")

            new_X.append(theta)
            new_y.append(mean_cost)
            new_feas.append(1.0)
            new_n_reps.append(len(costs_rep))

            # Update incumbent if better
            if mean_cost < state.incumbent_cost:
                print(f"  *** New incumbent! {mean_cost:.4f} < {state.incumbent_cost:.4f}")
                incumbent_idx = len(state.X_train) + cand_idx
                incumbent_cost = mean_cost
                incumbent_sd = sd_cost

        # Update state
        state = BOState(
            X_train=jnp.vstack([state.X_train, jnp.array(new_X)]),
            y_train=jnp.concatenate([state.y_train, jnp.array(new_y)]),
            feasible=jnp.concatenate([state.feasible, jnp.array(new_feas)]),
            n_reps=jnp.concatenate([state.n_reps, jnp.array(new_n_reps)]),
            incumbent_idx=state.incumbent_idx if 'incumbent_idx' not in locals() else incumbent_idx,
            incumbent_cost=incumbent_cost if 'incumbent_cost' in locals() else state.incumbent_cost,
            incumbent_sd=incumbent_sd if 'incumbent_sd' in locals() else state.incumbent_sd,
            iteration=iteration + 1,
        )

        # Save checkpoint
        if (iteration + 1) % config.save_every == 0:
            save_checkpoint(state, log_dir / f"checkpoint_{iteration+1:04d}.npz")

        # Log iteration
        log_iteration(state, log_dir / "bo_log.jsonl")

    print("\n=== Optimization complete ===")
    print(f"Best cost: {state.incumbent_cost:.4f}")
    print(f"Best parameters: {state.X_train[state.incumbent_idx]}")

    return state


def save_checkpoint(state: BOState, path: Path):
    """Save BO state to checkpoint."""
    jnp.savez(
        path,
        X_train=state.X_train,
        y_train=state.y_train,
        feasible=state.feasible,
        n_reps=state.n_reps,
        incumbent_idx=state.incumbent_idx,
        incumbent_cost=state.incumbent_cost,
        incumbent_sd=state.incumbent_sd,
        iteration=state.iteration,
    )
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: Path) -> BOState:
    """Load BO state from checkpoint."""
    data = jnp.load(path)
    return BOState(
        X_train=data["X_train"],
        y_train=data["y_train"],
        feasible=data["feasible"],
        n_reps=data["n_reps"],
        incumbent_idx=int(data["incumbent_idx"]),
        incumbent_cost=float(data["incumbent_cost"]),
        incumbent_sd=float(data["incumbent_sd"]),
        iteration=int(data["iteration"]),
    )


def log_iteration(state: BOState, log_path: Path):
    """Append iteration log to JSONL file."""
    log_entry = {
        "iteration": int(state.iteration),
        "incumbent_cost": float(state.incumbent_cost),
        "incumbent_sd": float(state.incumbent_sd),
        "n_evaluated": int(len(state.y_train)),
        "n_feasible": int(jnp.sum(state.feasible > 0.5)),
        "timestamp": datetime.now().isoformat(),
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
