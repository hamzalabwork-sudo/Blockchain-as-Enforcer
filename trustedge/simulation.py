"""Federated round engine: local training -> anomaly-based trust update ->
PoA2-gated, trust-weighted aggregation -> closed-loop quarantine
(Sec. III-C.3, III-E, Algorithm 1).

Implements the ablation cells of Table XVI as four aggregation "modes":

* "fedavg"           -- Eq. 8, unweighted average, no gate (vanilla FL / passive baseline)
* "authority_only"   -- hard pass/fail participation gate (statistical outlier check),
                        survivors weighted equally (identity-based Sybil resistance only)
* "association_only" -- Eq. 9-10 continuous trust-weighted average, no hard gate
                        (behavioral downweighting only)
* "poa2"             -- both: hard exclusion (Automatic Exclusion Rule) AND
                        continuous trust weighting (the full proposed mechanism)
* "krum" / "trimmed_mean" / "median" -- classic robust-aggregation baselines (Table IX)
* "centralized"      -- pools all client data and trains one model directly (no FL)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aggregation import coordinate_median, fedavg_aggregate, krum_select, trimmed_mean, trust_weighted_aggregate
from .attacks import apply_client_attack, gradient_reversal, gradient_scaling, random_update
from .datasets.base import FederatedDataset
from .models import build_model, clone_params, evaluate, load_flat_params, train_local
from .trust import TrustState, exclusion_rule


@dataclass
class FLClient:
    client_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    n_samples: int
    is_malicious: bool = False
    attack_type: str | None = None
    trust: TrustState = field(default_factory=TrustState)


def make_clients(dataset: FederatedDataset, binary: bool, malicious_ids: set[int] | None = None,
                  attack_assignment: dict[int, str] | None = None,
                  trust_beta: float = 0.7, trust_delta: float = 0.05) -> list[FLClient]:
    malicious_ids = malicious_ids or set()
    attack_assignment = attack_assignment or {}
    clients = []
    for cd in dataset.clients:
        y = (cd.y_train > 0).astype(np.int64) if binary else cd.y_train
        clients.append(FLClient(
            client_id=cd.client_id, X_train=cd.X_train, y_train=y, n_samples=len(cd.X_train),
            is_malicious=cd.client_id in malicious_ids,
            attack_type=attack_assignment.get(cd.client_id),
            trust=TrustState(beta=trust_beta, delta=trust_delta),
        ))
    return clients


def _detect_outliers(updates: list[np.ndarray], z_threshold: float = 1.5) -> list[bool]:
    """Statistical anomaly flag: True = flagged as (likely) malicious this round."""
    stacked = np.stack(updates)
    median = np.median(stacked, axis=0)
    dists = np.linalg.norm(stacked - median[None, :], axis=1)
    mean_d, std_d = dists.mean(), dists.std() + 1e-9
    z = (dists - mean_d) / std_d
    return (z > z_threshold).tolist()


@dataclass
class RoundLog:
    accuracy: float
    flagged_ids: list[int]
    excluded_ids: list[int]


def run_fl_experiment(
    dataset: FederatedDataset,
    model_name: str,
    method: str,
    binary: bool = True,
    rounds: int = 15,
    local_epochs: int = 1,
    lr: float = 0.01,
    malicious_ids: set[int] | None = None,
    attack_assignment: dict[int, str] | None = None,
    theta_t: float = 0.3,
    z_threshold: float = 1.5,
    n_byzantine_assumed: int | None = None,
    trust_beta: float = 0.7,
    trust_delta: float = 0.05,
    seed: int = 0,
) -> dict:
    malicious_ids = malicious_ids or set()
    clients = make_clients(dataset, binary, malicious_ids, attack_assignment, trust_beta, trust_delta)
    n_features = dataset.n_features
    n_classes = 2 if binary else dataset.n_classes
    X_test, y_test = dataset.global_test_set(binary=binary)
    rng = np.random.default_rng(seed)

    if method == "centralized":
        # Centralized training has no separate per-client gradient/update step,
        # so the gradient-level attacks (gradient_scaling/reversal/random_update)
        # have no attack surface here -- they'd silently become a no-op. Any
        # malicious client is instead modeled as contributing label-flipped
        # (data-poisoned) rows to the pooled training set, which is the
        # equivalent-strength attack a centralized pipeline is actually exposed to.
        Xs, ys = [], []
        for c in clients:
            X, y = c.X_train, c.y_train
            if c.is_malicious:
                X, y = apply_client_attack(X, y, n_classes, "label_flip", rng)
            Xs.append(X)
            ys.append(y)
        model = build_model(model_name, n_features, n_classes)
        train_local(model, np.concatenate(Xs), np.concatenate(ys), epochs=rounds, lr=lr, seed=seed)
        metrics = evaluate(model, X_test, y_test)
        return {"accuracy_curve": [metrics["accuracy"]], "final_metrics": metrics, "logs": []}

    global_model = build_model(model_name, n_features, n_classes)
    global_params = clone_params(global_model)
    n_byz = n_byzantine_assumed if n_byzantine_assumed is not None else max(1, int(round(len(malicious_ids))))

    accuracy_curve: list[float] = []
    logs: list[RoundLog] = []

    # Colluding "random_update" clients share ONE direction, fixed for the
    # whole run (not redrawn each round): independent per-client noise
    # self-cancels once averaged with the honest majority, and even a shared
    # direction re-drawn every round gets "healed" by honest re-training in
    # between (iterative FedAvg is naturally robust to non-persistent
    # perturbations) -- see attacks.random_update and the README for the
    # calibration this is based on (multiplier tuned so vanilla FedAvg's
    # accuracy drop lands near the paper's reported ~11pp at 20% poisoning).
    persistent_attack_direction = rng.normal(size=global_params.shape)
    persistent_attack_direction /= np.linalg.norm(persistent_attack_direction)
    RANDOM_UPDATE_MAGNITUDE = 13.0

    for t in range(rounds):
        raw_updates = []
        for c in clients:
            X, y = c.X_train, c.y_train
            if c.is_malicious and c.attack_type in ("label_flip",):
                X, y = apply_client_attack(X, y, n_classes, c.attack_type, rng)

            model = build_model(model_name, n_features, n_classes)
            load_flat_params(model, global_params)
            raw_updates.append(train_local(model, X, y, epochs=local_epochs, lr=lr, seed=seed + t + c.client_id))

        # Random-replacement magnitude is tied to THIS round's actual honest
        # update scale (adaptive across models/datasets/rounds).
        honest_deltas = [np.linalg.norm(u - global_params) for c, u in zip(clients, raw_updates) if not c.is_malicious]
        median_honest_delta = float(np.median(honest_deltas)) if honest_deltas else 1.0

        updates = []
        for c, update in zip(clients, raw_updates):
            if c.is_malicious and c.attack_type == "gradient_scaling":
                update = gradient_scaling(update, global_params)
            elif c.is_malicious and c.attack_type == "gradient_reversal":
                update = gradient_reversal(update, global_params)
            elif c.is_malicious and c.attack_type == "random_update":
                update = random_update(global_params, scale=median_honest_delta * RANDOM_UPDATE_MAGNITUDE, rng=rng,
                                        shared_direction=persistent_attack_direction)
            updates.append(update)

        flags = _detect_outliers(updates, z_threshold=z_threshold)
        flagged_ids = [c.client_id for c, f in zip(clients, flags) if f]
        for c, flagged in zip(clients, flags):
            c.trust.update(malicious_detected=flagged)

        if method == "fedavg":
            counts = [c.n_samples for c in clients]
            global_params = fedavg_aggregate(updates, counts)
            excluded_ids = []

        elif method == "authority_only":
            keep = [not f for f in flags]
            excluded_ids = [c.client_id for c, k in zip(clients, keep) if not k]
            kept_updates = [u for u, k in zip(updates, keep) if k] or updates
            kept_counts = [c.n_samples for c, k in zip(clients, keep) if k] or [c.n_samples for c in clients]
            global_params = fedavg_aggregate(kept_updates, kept_counts)

        elif method == "association_only":
            trust_scores = [c.trust.score for c in clients]
            counts = [c.n_samples for c in clients]
            global_params = trust_weighted_aggregate(updates, trust_scores, counts)
            excluded_ids = []

        elif method == "poa2":
            keep = [not exclusion_rule(c.trust.score, theta_t) for c in clients]
            excluded_ids = [c.client_id for c, k in zip(clients, keep) if not k]
            kept_updates = [u for u, k in zip(updates, keep) if k] or updates
            kept_trust = [c.trust.score for c, k in zip(clients, keep) if k] or [c.trust.score for c in clients]
            kept_counts = [c.n_samples for c, k in zip(clients, keep) if k] or [c.n_samples for c in clients]
            global_params = trust_weighted_aggregate(kept_updates, kept_trust, kept_counts)

        elif method == "krum":
            global_params = krum_select(updates, n_byz)
            excluded_ids = []

        elif method == "trimmed_mean":
            global_params = trimmed_mean(updates, trim_fraction=min(0.45, n_byz / len(updates)))
            excluded_ids = []

        elif method == "median":
            global_params = coordinate_median(updates)
            excluded_ids = []

        else:
            raise ValueError(method)

        eval_model = build_model(model_name, n_features, n_classes)
        load_flat_params(eval_model, global_params)
        acc = evaluate(eval_model, X_test, y_test)["accuracy"]
        accuracy_curve.append(acc)
        logs.append(RoundLog(accuracy=acc, flagged_ids=flagged_ids, excluded_ids=excluded_ids))

    final_model = build_model(model_name, n_features, n_classes)
    load_flat_params(final_model, global_params)
    final_metrics = evaluate(final_model, X_test, y_test)

    return {
        "accuracy_curve": accuracy_curve,
        "final_metrics": final_metrics,
        "logs": logs,
        "trust_scores": {c.client_id: c.trust.score for c in clients},
    }
