import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["BLIS_NUM_THREADS"] = "4"
import shutil

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import subprocess
import time
from utils import get_df_from_file

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "text.latex.preamble": r"\usepackage{amssymb} \usepackage{amsmath}",
    }
)

parser = argparse.ArgumentParser(
    description="Run analysis with specified run_dir and MASTER_SEED."
)
parser.add_argument("--run_dir", type=str, help="Directory containing the experiment data.")
parser.add_argument(
    "--MASTER_SEED", type=int, default=9850, help="Master seed for random number generation."
)
parser.add_argument(
    "--DEBUG", action="store_true", help="Run in debug mode with reduced computations."
)
args = parser.parse_args()

OUTPUT_BASE_DIR = f"plots_neurips_{args.MASTER_SEED}"
Z_NAMES = [r"$\mathfrak{z}_" + str(j + 1) + "$" for j in range(8)]

CONDITIONS = {
    "classic_cartpole": [
        (0.98, 0.5),
        (9.8, 0.5),
        (19.6, 0.5),
        (0.98, 0.8),
        (9.8, 0.8),
        (19.6, 0.8),
    ],
    "dmc_walker": [(0.98, 0.3), (0.98, 1.0), (9.81, 0.3), (9.81, 1.0), (19.6, 0.3), (19.6, 1.0)],
    "dmc_ball_in_cup": [
        (9.81, 0.09),
        (9.81, 0.15),
        (9.81, 0.3),
        (9.81, 0.54),
        (19.6, 0.09),
        (19.6, 0.15),
        (19.6, 0.3),
        (19.6, 0.54),
    ],
}

np.random.seed(args.MASTER_SEED)
ENV_MAPPING = {
    "classic_cartpole": "cartpole",
    "dmc_walker": "walker",
    "dmc_ball_in_cup": "ball_in_cup",
}
CONTEXT_LABELS = {
    "cartpole": ["Gravity", "Length"],
    "ball_in_cup": ["Gravity", "Distance"],
    "walker": ["Gravity", "Actuator Strength"],
}


def parse_folder_name(folder):
    carl_pos = folder.find("carl_")
    if carl_pos == -1:
        raise ValueError(f"Invalid folder name: {folder}")
    start = carl_pos + len("carl_")
    end = folder.find("_double_box", start)
    env = folder[start:end]
    enc_pos = folder.find("_enc_", end)
    dec_pos = folder.find("_dec_", enc_pos)
    modality = folder[enc_pos + len("_enc_") : dec_pos]
    ctx_pos = folder.find("_ctxencoder_", dec_pos)
    approach = folder[ctx_pos + len("_ctxencoder_") :]
    return env, modality, approach


def plot_tsne(ax, zs, labels, title):
    uniq_labels = np.unique(labels, axis=0)
    cmap = plt.get_cmap("tab20")
    for j, lab in enumerate(uniq_labels):
        idxs = (labels == lab).all(axis=-1)
        color_j = j * cmap.N // len(uniq_labels)
        ax.scatter(zs[idxs, 0], zs[idxs, 1], color=cmap(color_j), s=4)
        xtext, ytext = np.median(zs[idxs, :], axis=0)
        short_name = ", ".join([f"{l:.2f}" for l in lab])
        txt = ax.text(xtext, ytext, short_name, fontsize=8)
        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
        )
    ax.set_axis_off()
    ax.set_title(title)


def tsne_grid_analysis(df, env, modality, approach, output_dir):
    if env not in CONDITIONS:
        print(f"No conditions defined for {env}, skipping t-SNE analysis.")
        return

    df_filtered = df[df["real_context"].apply(tuple).isin(CONDITIONS[env]) & (df["episode"] == 1)]
    df_filtered = df_filtered.sample(frac=0.5, random_state=args.MASTER_SEED)
    if df_filtered.empty:
        print(f"No data after filtering for {env}, skipping t-SNE analysis.")
        return

    labels = np.stack(df_filtered["real_context"].to_numpy())
    if "embed" in df_filtered.columns:
        first_col = ("embedding", np.stack(df_filtered["embed"].to_numpy()))
    else:
        first_col = (r"$o_t$", np.stack(df_filtered["obs"].to_numpy()))
    combinations = [
        first_col,
        (r"$z_t$", np.stack(df_filtered["posterior"].to_numpy())),
        (
            r"$z_t \oplus a_t$",
            np.concatenate(
                [
                    np.stack(df_filtered["posterior"].to_numpy()),
                    np.stack(df_filtered["action"].to_numpy()),
                ],
                axis=-1,
            ),
        ),
        (r"$a_t$", np.stack(df_filtered["action"].to_numpy())),
        (r"$\hat{z}_t$", np.stack(df_filtered["prior"].to_numpy())),
        (
            r"$\hat{z}_t \oplus a_t$",
            np.concatenate(
                [
                    np.stack(df_filtered["prior"].to_numpy()),
                    np.stack(df_filtered["action"].to_numpy()),
                ],
                axis=-1,
            ),
        ),
    ]
    if "context_representation" in df_filtered.columns:
        combinations.extend(
            [
                (
                    r"$\mathfrak{z}_t \oplus a_t$",
                    np.concatenate(
                        [
                            np.stack(df_filtered["context_representation"].to_numpy()),
                            np.stack(df_filtered["action"].to_numpy()),
                        ],
                        axis=-1,
                    ),
                ),
                (
                    r"$\hat{z}_t \oplus a_t \oplus \mathfrak{z}_t$",
                    np.concatenate(
                        [
                            np.stack(df_filtered["prior"].to_numpy()),
                            np.stack(df_filtered["action"].to_numpy()),
                            np.stack(df_filtered["context_representation"].to_numpy()),
                        ],
                        axis=-1,
                    ),
                ),
            ]
        )
    zs_list = Parallel(n_jobs=-1)(
        delayed(
            lambda rep: TSNE(
                n_components=2, random_state=args.MASTER_SEED, perplexity=30
            ).fit_transform(rep)
        )(rep)
        for _, rep in combinations
    )
    zs_list = [(name, zs) for (name, _), zs in zip(combinations, zs_list)]
    n_cols = (len(combinations) + 1) // 2
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 6))
    axes = axes.flatten()
    for i, (name, zs) in enumerate(zs_list):
        plot_tsne(axes[i], zs, labels, name)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{env} {modality} {approach}")
    fig.savefig(os.path.join(output_dir, "tsne_grid.pdf"), bbox_inches="tight")
    plt.close(fig)



def ablation_auc_analysis(data_path, env, modality, approach, output_dir):
    """Compute AUC for each context dimension and analyze significance with an ensemble."""
    base_seed = args.MASTER_SEED + hash(f"{env}_{modality}_{approach}") % (2**32)
    np.random.seed(base_seed)

    try:
        data = np.load(data_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return None

    n_trajs = 200 if args.DEBUG else 2000

    def compute_auc_for_dim(dim, data, base_seed):
        """Compute AUC for a given context dimension using an ensemble with cross-validation."""
        np.random.seed(base_seed + dim)

        # Select perturbed z dim trajectories
        key = f"z_dim{dim}"
        original_trajs = data["original"][key][:n_trajs]
        imagined_trajs = data["imagined"][key][:n_trajs]

        X_seq = np.concatenate([original_trajs, imagined_trajs], axis=0)
        y = np.concatenate([np.zeros(len(original_trajs)), np.ones(len(imagined_trajs))])
        perm = np.random.permutation(len(y))
        X_seq = X_seq[perm]
        y = y[perm]

        X_flat = X_seq.reshape(X_seq.shape[0], -1)

        # Define number of estimators based on debug mode
        if args.DEBUG:
            ada_estimators = 50
        else:
            ada_estimators = 200

        # Define ensemble of strong classifiers from scikit-learn
        classifiers = [
            SVC(
                kernel='rbf',
                C=30,
                probability=True,
                max_iter=50 if args.DEBUG else -1,
                random_state=base_seed + dim
            ),
            MLPClassifier(
                hidden_layer_sizes=(1024, 1024),
                alpha=0.001,
                max_iter=50 if args.DEBUG else 2000,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=base_seed + dim
            ),
            AdaBoostClassifier(
                n_estimators=ada_estimators,
                random_state=base_seed + dim
            )
        ]

        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=base_seed + dim)
        all_probs = []
        all_y = []

        for train_idx, test_idx in skf.split(X_flat, y):
            X_train, X_test = X_flat[train_idx], X_flat[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train ensemble and average probabilities
            probs_list = []
            for clf in classifiers:
                clf.fit(X_train_scaled, y_train)
                probs = clf.predict_proba(X_test_scaled)[:, 1]
                probs_list.append(probs)
            # Print AUC for each classifier in the ensemble for debugging
            print("AUC PER METHOD", [roc_auc_score(y_test, probs) for probs in probs_list])
            probs = np.mean(probs_list, axis=0)
            all_probs.extend(probs)
            all_y.extend(y_test)

        # Convert to numpy arrays for efficiency
        all_probs = np.array(all_probs)
        all_y = np.array(all_y)

        # Compute overall AUC
        auc = roc_auc_score(all_y, all_probs)

        # Bootstrap for confidence intervals
        bootstrap_aucs = []
        n_samples = len(all_y)
        bootstrap_len = 100 if args.DEBUG else 500
        for _ in range(bootstrap_len):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            sampled_probs = all_probs[indices]
            sampled_y = all_y[indices]
            if len(np.unique(sampled_y)) > 1:
                bootstrap_aucs.append(roc_auc_score(sampled_y, sampled_probs))

        bootstrap_aucs = np.array(bootstrap_aucs)
        mean_auc = np.mean(bootstrap_aucs)
        ci_lower = np.percentile(bootstrap_aucs, 2.5)
        ci_upper = np.percentile(bootstrap_aucs, 97.5)

        return dim, mean_auc, ci_lower, ci_upper, bootstrap_aucs, all_probs, all_y

    # Compute AUC for each context in parallel
    results = Parallel(n_jobs=-1)(delayed(compute_auc_for_dim)(dim, data, base_seed) for dim in range(8))
    # results = [compute_auc_for_dim(dim, data, base_seed) for dim in range(8)]

    # Organize results
    dim_results = []
    for dim, mean_auc, ci_lower, ci_upper, bootstrap_aucs, probs, y_val in results:
        dim_results.append(
            {
                "c_dim": dim,
                "mean_auc": mean_auc,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "bootstrap_aucs": bootstrap_aucs,
                "probs": probs,
                "labels": y_val,
            }
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    mean_aucs = [res["mean_auc"] for res in dim_results]
    err_lower = [res["mean_auc"] - res["ci_lower"] for res in dim_results]
    err_upper = [res["ci_upper"] - res["mean_auc"] for res in dim_results]
    x_pos = range(len(dim_results))

    ax.bar(
        x_pos,
        mean_aucs,
        yerr=[err_lower, err_upper],
        capsize=5,
        color="skyblue",
        edgecolor="black",
        alpha=0.8,
    )
    baseline = np.mean(mean_aucs)
    ax.axhline(baseline, color='red', linestyle='--', linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(Z_NAMES, fontsize=20, rotation=45, ha='right')
    ax.set_ylabel("AUC ± 95% CI", fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "auc_per_context.pdf"))
    plt.close()
    return np.array([res["mean_auc"] for res in dim_results])


def counterfactual_obs(data_path, top_dim, cf_dir):
    for ctx_id in range(2):
        result = subprocess.run(
            [
                "uv",
                "run",
                "-m",
                "contextual_mbrl.dreamer.record_counterfactual_plausibility_obs",
                "--logdir",
                f"./logs_dali/{data_path}/71",
                "--jax.platform",
                "cpu",
                "--ctx_id",
                str(ctx_id),
                "--z_dim",
                str(top_dim),
            ],
            capture_output=True,
            text=True,
            check=True,  # It's good practice to check for errors
            cwd="..",
        )
        print(result.stdout)
        if result.stderr:
            print("Error:", result.stderr)

    source_path = os.path.join("..", f"logs_dali/{data_path}/71", "obs_counterfactuals")

    os.makedirs(cf_dir, exist_ok=True)

    destination_name = "obs_counterfactuals"
    final_path_at_destination = os.path.join(cf_dir, destination_name)

    if os.path.exists(source_path):
        shutil.move(source_path, final_path_at_destination)

def counterfactual_imgs(data_path, top_dim, cf_dir):
    for ctx_id in range(2):
        result = subprocess.run(
            [
                "uv",
                "run",
                "-m",
                "contextual_mbrl.dreamer.record_counterfactual_plausibility",
                "--logdir",
                f"./logs_dali/{data_path.replace('obs', 'img')}/71",
                "--jax.platform",
                "cpu",
                "--ctx_id",
                str(ctx_id),
                "--z_dim",
                str(top_dim),
            ],
            capture_output=True,
            text=True,
            check=True,  # It's good practice to check for errors
            cwd="..",
        )
        print(result.stdout)
        if result.stderr:
            print("Error:", result.stderr)

    source_path = os.path.join("..", f"logs_dali/{data_path.replace('obs', 'img')}/71", "img_counterfactuals")
    os.makedirs(cf_dir, exist_ok=True)
    destination_name = "img_counterfactuals"
    final_path_at_destination = os.path.join(cf_dir, destination_name)

    if os.path.exists(source_path):
        shutil.move(source_path, final_path_at_destination)




def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    folder = args.run_dir
    env, modality, approach = parse_folder_name(folder)
    env_short = ENV_MAPPING.get(env, "unknown")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    if env_short == "unknown":
        print(f"Skipping unknown environment: {folder}")
        return
    print(f"\nProcessing {env_short} - {modality} - {approach}")
    pickle_path = f"../logs_dali/{folder}/71/carl_{env}_ctx_represenations_windows.pkl"
    try:
        df = get_df_from_file(pickle_path)
        X = np.stack(df["context_representation"].to_numpy())
        y = np.stack(df["real_context"].to_numpy())
        idx = np.random.choice(len(X), size=2000 if args.DEBUG else 10000, replace=False)
        X, y = X[idx], y[idx]
    except Exception as e:
        print(f"Error loading {pickle_path}: {e}")
        return

    combination_dir = f"{env_short}_{modality}_{approach}"
    auc_dir = os.path.join(OUTPUT_BASE_DIR, "ablation_auc", combination_dir)
    tsne_dir = os.path.join(OUTPUT_BASE_DIR, "tsne_grid_plots", combination_dir)
    cf_dir = os.path.join(OUTPUT_BASE_DIR, "countf_plots", combination_dir)
    for d in [auc_dir, tsne_dir, cf_dir]:
        os.makedirs(d, exist_ok=True)

    trajectory_dataset_path = f"../logs_dali/{folder}/71/dataset_ep200_sigma.pkl"
    start_time = time.time()
    if os.path.exists(trajectory_dataset_path):
        print(f"Ablation data found at {trajectory_dataset_path}")
        auc_values = ablation_auc_analysis(
            trajectory_dataset_path, env_short, modality, approach, auc_dir
        )
    else:
        print(f"Ablation data not found at {trajectory_dataset_path}")
        auc_values = np.zeros((8,))
    print(f"Time taken for ablation_auc_analysis: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    tsne_grid_analysis(df, env, modality, approach, tsne_dir)
    print(f"Time taken for tsne_grid_analysis: {time.time() - start_time:.2f} seconds")

    top_dim = auc_values.argmax().tolist()
    print(f"\n{'=' * 30}\nTop dimension index: {top_dim}\n{'=' * 30}\n")

    start_time = time.time()
    counterfactual_obs(folder, top_dim, cf_dir)
    print(f"Time taken for counterfactual obs: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    counterfactual_imgs(folder, top_dim, cf_dir)
    print(f"Time taken for counterfactual img: {time.time() - start_time:.2f} seconds")

    print(
        f"Completed {env_short} - {modality} - {approach} in {time.time() - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
