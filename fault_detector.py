"""
fault_detector.py — ML Fault Detection for Plasma Etch Tool
============================================================
Trains a Random Forest classifier on extracted sensor features
to detect and classify equipment faults in a plasma etch tool.

Models trained:
  • Random Forest (primary) — interpretable, industry-preferred
  • Gradient Boosting       — higher accuracy comparison
  • Logistic Regression     — linear baseline

Output:
  • Classification report (precision, recall, F1 per fault class)
  • Confusion matrix
  • Feature importance ranking
  • Saved model: fault_detector_model.pkl

Usage:
  python fault_detector.py              # train and evaluate
  python fault_detector.py --save       # save model to disk
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
FAULT_CLASSES = {
    0: "Normal",
    1: "Pressure Drift",
    2: "RF Instability",
    3: "Gas Flow Anomaly",
    4: "Chuck Temp Excursion",
}

COLORS = {
    0: "#27AE60",   # green — normal
    1: "#E74C3C",   # red — pressure
    2: "#E67E22",   # orange — RF
    3: "#9B59B6",   # purple — gas
    4: "#3498DB",   # blue — chuck
}

C_BG   = "#0E1117"
C_BLUE = "#1F3A6B"
SEPARATOR = "=" * 68


# ── Load and prepare data ─────────────────────────────────────────────────────
def load_data(path="etch_features.csv"):
    df = pd.read_csv(path)
    drop_cols = ["fault_class", "fault_label", "sample_idx"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values
    y = df["fault_class"].values
    return X, y, feature_cols


# ── Train models ──────────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
    }

    trained = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline

    return trained


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=[FAULT_CLASSES[i] for i in range(5)],
        output_dict=True
    )
    return {
        "y_pred": y_pred, "y_prob": y_prob,
        "accuracy": acc, "f1": f1,
        "confusion_matrix": cm, "report": report
    }


# ── Feature importance ────────────────────────────────────────────────────────
def get_feature_importance(rf_pipeline, feature_cols, top_n=20):
    rf = rf_pipeline.named_steps["clf"]
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    return [(feature_cols[i], importances[i]) for i in idx]


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    labels = [FAULT_CLASSES[i] for i in range(5)]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(labels, rotation=30, ha="right",
                       color="white", fontsize=9)
    ax.set_yticklabels(labels, color="white", fontsize=9)

    for i in range(5):
        for j in range(5):
            val   = cm[i, j]
            pct   = cm_norm[i, j]
            color = "white" if pct > 0.5 else "#AAAAAA"
            ax.text(j, i, f"{val}\n({pct:.0%})",
                    ha="center", va="center",
                    color=color, fontsize=8, fontweight="bold")

    ax.set_xlabel("Predicted", color="white", fontsize=11)
    ax.set_ylabel("Actual",    color="white", fontsize=11)
    ax.set_title("Confusion Matrix — Random Forest Fault Detector",
                 color="white", fontsize=12, fontweight="bold", pad=15)
    ax.tick_params(colors="white")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"   Saved → {save_path}")
    return fig


def plot_feature_importance(importances, save_path=None):
    names  = [x[0] for x in importances]
    values = [x[1] for x in importances]

    # Color bars by sensor
    sensor_colors = {
        "rf_power":        "#E67E22",
        "pressure":        "#E74C3C",
        "gas_flow":        "#9B59B6",
        "chuck_temp":      "#3498DB",
        "bias_voltage":    "#27AE60",
        "reflected_power": "#F39C12",
        "dc_bias":         "#1ABC9C",
    }

    bar_colors = []
    for name in names:
        color = "#AAAAAA"
        for sensor, c in sensor_colors.items():
            if sensor in name:
                color = c
                break
        bar_colors.append(color)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    bars = ax.barh(range(len(names)), values[::-1],
                   color=bar_colors[::-1], alpha=0.85, height=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names[::-1]],
                       color="white", fontsize=9)
    ax.set_xlabel("Feature Importance", color="white", fontsize=11)
    ax.set_title("Top 20 Features — Random Forest Fault Detector\n"
                 "(higher = more predictive of fault class)",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#4A5568")

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=s.replace("_", " ").title())
        for s, c in sensor_colors.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor="#0D1117", edgecolor="#4A5568",
              labelcolor="white", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"   Saved → {save_path}")
    return fig


def plot_model_comparison(results, save_path=None):
    """Bar chart comparing accuracy and F1 across models."""
    models  = list(results.keys())
    accs    = [results[m]["accuracy"] * 100 for m in models]
    f1s     = [results[m]["f1"] * 100 for m in models]

    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy",
                   color="#1F3A6B", alpha=0.9)
    bars2 = ax.bar(x + w/2, f1s,  w, label="Weighted F1",
                   color="#27AE60", alpha=0.9)

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom",
                color="white", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, color="white", fontsize=10)
    ax.set_ylabel("Score (%)", color="white", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Model Comparison — Accuracy vs F1 Score",
                 color="white", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#0D1117", edgecolor="#4A5568",
              labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4A5568")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"   Saved → {save_path}")
    return fig


# ── Text report ───────────────────────────────────────────────────────────────
def print_header(text):
    print(f"\n{SEPARATOR}")
    print(f"  {text}")
    print(SEPARATOR)

def print_section(text):
    print(f"\n  ── {text}")

def print_results(results, model_name):
    res = results[model_name]
    print_header(f"{model_name.upper()} — RESULTS")
    print(f"  Accuracy : {res['accuracy']:.4f}  ({res['accuracy']:.2%})")
    print(f"  F1 Score : {res['f1']:.4f}  ({res['f1']:.2%})")

    print_section("Per-Class Performance")
    report = res["report"]
    print(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*65}")
    for cls_name in [FAULT_CLASSES[i] for i in range(5)]:
        if cls_name in report:
            r = report[cls_name]
            flag = "  ✓" if r["f1-score"] >= 0.95 else "  ⚠" if r["f1-score"] >= 0.85 else "  ✗"
            print(f"  {cls_name:<25} {r['precision']:>10.3f} "
                  f"{r['recall']:>10.3f} {r['f1-score']:>10.3f} "
                  f"{int(r['support']):>10}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ML fault detector for plasma etch tool sensor data.")
    parser.add_argument("--save",   action="store_true",
                        help="Save charts and model to disk")
    parser.add_argument("--outdir", default=".",
                        help="Output directory for charts")
    args = parser.parse_args()

    if args.save:
        os.makedirs(args.outdir, exist_ok=True)

    # ── Load data ──
    print("\nLoading sensor feature data...")
    X, y, feature_cols = load_data()
    print(f"  {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  {len(set(y))} fault classes")

    # ── Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Train ──
    print_header("TRAINING MODELS")
    models = train_models(X_train, y_train)
    for name in models:
        print(f"  ✓ {name} trained")

    # ── Cross-validation ──
    print_section("5-Fold Cross-Validation (Random Forest)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_scores = cross_val_score(models["Random Forest"], X, y,
                                cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    print(f"  Folds: {[f'{s:.3f}' for s in rf_scores]}")

    # ── Evaluate all models ──
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)

    # ── Print reports ──
    for name in models:
        print_results(results, name)

    # ── Feature importance ──
    importances = get_feature_importance(
        models["Random Forest"], feature_cols, top_n=20)

    print_header("TOP 10 MOST IMPORTANT FEATURES")
    for i, (feat, imp) in enumerate(importances[:10]):
        bar = "█" * int(imp * 200)
        print(f"  {i+1:2}. {feat:<35} {imp:.4f}  {bar}")

    # ── Save charts ──
    if args.save:
        print_header("SAVING CHARTS")

        plot_confusion_matrix(
            results["Random Forest"]["confusion_matrix"],
            save_path=os.path.join(args.outdir, "fault_confusion_matrix.png"))

        plot_feature_importance(
            importances,
            save_path=os.path.join(args.outdir, "fault_feature_importance.png"))

        plot_model_comparison(
            results,
            save_path=os.path.join(args.outdir, "fault_model_comparison.png"))

        # Save model
        model_path = os.path.join(args.outdir, "fault_detector_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model":        models["Random Forest"],
                "feature_cols": feature_cols,
                "fault_classes": FAULT_CLASSES,
            }, f)
        print(f"   Model saved → {model_path}")

    print(f"\n{SEPARATOR}")
    print("  Training complete.")
    print(f"{SEPARATOR}\n")

    return models, results, feature_cols, importances


if __name__ == "__main__":
    main()
