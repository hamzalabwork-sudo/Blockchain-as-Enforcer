"""Reproduces Sec. IV-C (Table IV/V/VI): CNN vs LSTM vs BiLSTM detection
performance, federated convergence, and per-attack-type detection rate,
across all four IIoT datasets, no adversarial clients.

Generates:
  results/detection_performance.csv   -- Table IV analogue (binary + multiclass)
  results/convergence.csv             -- Table V analogue
  results/detection_rate_per_attack.csv -- Table VI analogue
  results/detection_convergence.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score

from experiments.plotting import apply_style, method_color
from trustedge.datasets.iotforge import load_iotforge
from trustedge.datasets.iot_cad import load_iot_cad
from trustedge.datasets.ton_iot import load_ton_iot
from trustedge.datasets.wustl_iiot import load_wustl
from trustedge.models import build_model, train_local
from trustedge.simulation import run_fl_experiment

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATASETS = {
    "IoTForge Pro": load_iotforge,
    "ToN-IoT": load_ton_iot,
    "IoT-CAD": load_iot_cad,
    "WUSTL-IIoT-2021": load_wustl,
}
MODELS = ["CNN", "LSTM", "BiLSTM"]
ROUNDS = 15


def convergence_round(acc_curve: list[float], tolerance: float = 0.98) -> int:
    arr = np.array(acc_curve)
    target = tolerance * arr[-1]
    hits = np.where(arr >= target)[0]
    return int(hits[0] + 1) if len(hits) else len(arr)


def per_attack_recall(dataset, model_name: str, seed: int = 0) -> dict:
    clients = dataset.clients
    X_train = np.concatenate([c.X_train for c in clients])
    y_train = np.concatenate([c.y_train for c in clients])
    X_test, y_test = dataset.global_test_set(binary=False)

    model = build_model(model_name, dataset.n_features, dataset.n_classes)
    train_local(model, X_train, y_train, epochs=ROUNDS, lr=0.01, seed=seed)
    with torch.no_grad():
        preds = model(torch.as_tensor(X_test, dtype=torch.float32)).argmax(1).numpy()

    recalls = {}
    for cls_idx, name in enumerate(["Benign"] + dataset.attack_names):
        mask = y_test == cls_idx
        if mask.sum() == 0 or cls_idx == 0:
            continue
        recalls[name] = float(recall_score(y_test == cls_idx, preds == cls_idx))
    return recalls


def main(rounds: int = ROUNDS, seed: int = 0) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    apply_style()

    perf_rows, conv_rows, det_rows = [], [], []
    convergence_curves: dict[str, dict[str, list[float]]] = {}

    for ds_name, loader in DATASETS.items():
        dataset = loader(n_clients=10, samples_per_client=300, seed=seed)
        convergence_curves[ds_name] = {}

        for model_name in MODELS:
            for binary in (True, False):
                res = run_fl_experiment(dataset, model_name, "fedavg", binary=binary, rounds=rounds, lr=0.01, seed=seed)
                m = res["final_metrics"]
                perf_rows.append([ds_name, model_name, "binary" if binary else "multiclass",
                                   f"{m['accuracy']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                                   f"{m['f1']:.4f}", f"{m['fpr']:.4f}" if binary else "N/A"])
                if binary:
                    conv_rows.append([ds_name, model_name, convergence_round(res["accuracy_curve"]),
                                       f"{1 - m['accuracy']:.4f}"])
                    convergence_curves[ds_name][model_name] = res["accuracy_curve"]
                    print(f"{ds_name:20s} {model_name:8s} binary acc={m['accuracy']:.4f} conv_round={conv_rows[-1][2]}")

        for model_name in MODELS:
            recalls = per_attack_recall(dataset, model_name, seed=seed)
            for attack, r in recalls.items():
                det_rows.append([ds_name, attack, model_name, f"{r*100:.2f}"])

    with open(RESULTS_DIR / "detection_performance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Model", "Task", "Accuracy", "Precision", "Recall", "F1", "FPR"])
        w.writerows(perf_rows)

    with open(RESULTS_DIR / "convergence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Model", "Rounds to Converge", "Final Error (1-Acc)"])
        w.writerows(conv_rows)

    with open(RESULTS_DIR / "detection_rate_per_attack.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Attack Type", "Model", "Detection Rate (%)"])
        w.writerows(det_rows)

    # --- convergence plot (one dataset, all 3 models) ------------------------------------
    example_ds = "WUSTL-IIoT-2021"
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for model_name in MODELS:
        curve = convergence_curves[example_ds][model_name]
        ax.plot(range(1, rounds + 1), curve, label=model_name, color=method_color(model_name))
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"{example_ds}: Federated Convergence (binary)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "detection_convergence.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved tables and plot to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
