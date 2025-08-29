import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import sys

sys.path.append("../")
from contextual_mbrl.dreamer.envs import _TASK2CONTEXTS, _TASK2ENV

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "text.latex.preamble": r"\usepackage{amssymb} \usepackage{amsmath} \DeclareRobustCommand{\rchi}{{\mathpalette\irchi\relax}} \newcommand{\irchi}[2]{\raisebox{\depth}{$#1\chi$}}",
    }
)

# Configuration
DEFAULT_TASKS = ["dmc_ball_in_cup", "dmc_walker"]
DEFAULT_MODALITIES = ["obs", "img"]
DEFAULT_METRICS = ["Median", "IQM", "Mean", "Optimality Gap"]

T_STEP = "normalized"
TRAIN_SETTINGS = ["single", "double_box"]

# Mappings
TASK_MAP = {
    "dmc_walker": "dmc_Walker",
    "dmc_ball_in_cup": "dmc_Ball_in_Cup",
}
TYPE_MAP = {"Interp": "Interpolation", "Extrap": "Extrapolation", "Inter_Extra": "Mixed Regime"}
CTX_ALG_MAP = {
    "no_ctx": "Dreamer-DR",
    "enc_dec_ctx": "cRSSM-S",
    "pgm_ctx": "cRSSM-D",
    "no_ctx_ctxtransformer": "DALI-S",
    "no_ctx_ctxtransformer_coupling": r"DALI-S-$\rchi$",
}
MODALITY_MAP = {"obs": "Featurized", "img": "Pixel"}
TASK2MAXRETURN = { "dmc_walker": 1000, "dmc_ball_in_cup": 1000}

# Color mapping for algorithms
COLOR_MAP = {
    "Dreamer-DR": "#0072B2",  # Blue
    "DALI-S": "#009E73",  # Green
    r"DALI-S-$\rchi$": "#E69F00",  # Yellow
    "cRSSM-S": "#CC79A7",  # Pink
    "cRSSM-D": "#D55E00",  # Vermillion
}

# Metric Aggregation Map
METRICS_AGG_MAP = {
    "Median": metrics.aggregate_median,
    "IQM": metrics.aggregate_iqm,
    "Mean": metrics.aggregate_mean,
    "Optimality Gap": metrics.aggregate_optimality_gap,
}


def compute_agg_all(
    log_dir,
    tasks,
    train_settings,
    modality,
    best_metrics,
    random_metrics,
    algorithms,
    target_metrics,
    reference_alg,
):
    scores_per_alg = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        for setting in train_settings:
            context_ids = [0, 1] if setting == "single" else [None]
            for ctx_id in context_ids:
                for alg in algorithms:
                    i_score, e_score, ie_score = get_ies_scores(
                        log_dir, task, setting, ctx_id, modality, alg, best_metrics, random_metrics
                    )
                    mapped_alg = CTX_ALG_MAP[alg]
                    scores_per_alg["Interp"][mapped_alg].append(i_score)
                    scores_per_alg["Extrap"][mapped_alg].append(e_score)
                    scores_per_alg["Inter_Extra"][mapped_alg].append(ie_score)

    final_scores_dict = defaultdict(dict)
    concatenated_scores = defaultdict(dict)
    for alg in [CTX_ALG_MAP[alg] for alg in algorithms]:
        for type in ["Interp", "Extrap", "Inter_Extra"]:
            scores = scores_per_alg[type][alg]
            min_seeds = min([s.shape[-1] for s in scores if s.size != 0], default=0)
            score_list = [s[:, :min_seeds] for s in scores if s.size != 0]
            if score_list:
                concatenated = np.concatenate(score_list, axis=0).T
                concatenated_scores[type][alg] = concatenated
                agg_func = lambda x: np.array([METRICS_AGG_MAP[m](x) for m in target_metrics])
                agg_scores, agg_cis = rly.get_interval_estimates(
                    {alg: concatenated}, agg_func, reps=100
                )
                for i, metric in enumerate(target_metrics):
                    final_scores_dict[alg][f"{type}_{metric}"] = np.round(agg_scores[alg][i], 3)
                    final_scores_dict[alg][f"{type}_{metric}_ci"] = agg_cis[alg][:, i]

    # Compute Probability of Improvement
    reference_alg_display = reference_alg
    for type in ["Interp", "Extrap", "Inter_Extra"]:
        all_pairs = {}
        for alg in [a for a in concatenated_scores[type] if a != reference_alg_display]:
            pair_name = f"{reference_alg_display}_{alg}"
            if (
                reference_alg_display in concatenated_scores[type]
                and alg in concatenated_scores[type]
            ):
                all_pairs[pair_name] = (
                    concatenated_scores[type][reference_alg_display],
                    concatenated_scores[type][alg],
                )
        if all_pairs:
            probabilities, probability_cis = rly.get_interval_estimates(
                all_pairs, metrics.probability_of_improvement, reps=100
            )
            for alg in [a for a in concatenated_scores[type] if a != reference_alg_display]:
                pair_name = f"{reference_alg_display}_{alg}"
                if pair_name in probabilities:
                    final_scores_dict[alg][f"{type}_IP"] = np.round(probabilities[pair_name], 3)
                    final_scores_dict[alg][f"{type}_IP_ci"] = probability_cis[pair_name]

    return pd.DataFrame(final_scores_dict).T


def get_expert_and_random_metrics(logdir):
    expert_metrics = {task: defaultdict(list) for task in DEFAULT_TASKS}
    random_metrics = {task: defaultdict(list) for task in DEFAULT_TASKS}
    for task_name in DEFAULT_TASKS:
        for exp_dir in (logdir / f"carl_{task_name}" / "expert").glob("*_specific_*"):
            expert_returns = []
            random_returns = []
            for seed_dir in exp_dir.iterdir():
                if (seed_dir / "eval.jsonl").exists():
                    metrics = json.loads((seed_dir / "eval.jsonl").read_text().split("\n")[0])
                    expert_returns.append(
                        TASK2MAXRETURN[task_name]
                        if task_name in ["dmc_ball_in_cup"]
                        else metrics["return"]
                    )
                if (seed_dir / "eval_random_policy.jsonl").exists():
                    metrics = json.loads(
                        (seed_dir / "eval_random_policy.jsonl").read_text().split("\n")[0]
                    )
                    random_returns.append(metrics["return"])
            if not expert_returns:
                continue
            ctx_0 = _TASK2CONTEXTS[task_name][0]["context"]
            ctx_1 = _TASK2CONTEXTS[task_name][1]["context"]
            env = _TASK2ENV[task_name]
            default_0, default_1 = (
                env.get_default_context()[ctx_0],
                env.get_default_context()[ctx_1],
            )
            ctx_k_v_str = exp_dir.name.split("specific_")[1].split("_enc")[0]
            ctx_val = [default_0, default_1]
            for k_v in ctx_k_v_str.split("_"):
                k, v = k_v.split("-")
                ctx_val[int(k)] = float(v)
            expert_metrics[task_name][(ctx_val[0], ctx_val[1])] = np.max(expert_returns)
            random_metrics[task_name][(ctx_val[0], ctx_val[1])] = np.max(random_returns)
    return expert_metrics, random_metrics


def match_and_normalize_metrics(agent_metrics, expert_metrics, random_metrics):
    normalized_metrics = {}
    min_ctx_0, max_ctx_0 = min(k[0] for k in agent_metrics), max(k[0] for k in agent_metrics)
    min_ctx_1, max_ctx_1 = min(k[1] for k in agent_metrics), max(k[1] for k in agent_metrics)
    norm_best_keys = [
        (
            (k[0] - min_ctx_0) / (max_ctx_0 - min_ctx_0 + 1e-6),
            (k[1] - min_ctx_1) / (max_ctx_1 - min_ctx_1 + 1e-6),
        )
        for k in expert_metrics
    ]
    best_keys = list(expert_metrics.keys())
    for current_key, returns in agent_metrics.items():
        norm_current_key = (
            (current_key[0] - min_ctx_0) / (max_ctx_0 - min_ctx_0 + 1e-6),
            (current_key[1] - min_ctx_1) / (max_ctx_1 - min_ctx_1 + 1e-6),
        )
        distances = cdist([norm_current_key], norm_best_keys)
        nearest_key = best_keys[np.argmin(distances)]
        normalized_metrics[current_key] = (returns - random_metrics[nearest_key]) / (
            expert_metrics[nearest_key] - random_metrics[nearest_key]
        )
    return normalized_metrics


def get_ies_scores(
    log_dir, task, train_setting, context_idx, modality, ctx_type, best_metrics, random_metrics
):
    train_setting_full = (
        f"{train_setting}_{context_idx}" if train_setting == "single" else train_setting
    )
    modality_dir = "featurized" if modality == "obs" else "pixel"
    exp_paths = {
        "no_ctx": f"carl_{task}_{train_setting_full}_enc_{modality}_dec_{modality}_{T_STEP}",
        "enc_dec_ctx": f"carl_{task}_{train_setting_full}_enc_{modality}_ctx_dec_{modality}_ctx_{T_STEP}",
        "pgm_ctx": f"carl_{task}_{train_setting_full}_enc_{modality}_dec_{modality}_pgm_ctx_{T_STEP}",
        "no_ctx_ctxtransformer": f"carl_{task}_{train_setting_full}_enc_{modality}_dec_{modality}_ctxencoder_transformer_{T_STEP}",
        "no_ctx_ctxtransformer_crossmodal": f"carl_{task}_{train_setting_full}_enc_{modality}_dec_{modality}_ctxencoder_transformer_{T_STEP}_crossmodal",
    }
    exp_path = log_dir / f"carl_{task}" / modality_dir / exp_paths.get(ctx_type, "")
    if not exp_path.exists():
        print(f"Path does not exist: {exp_path}")
        return np.array([]), np.array([]), np.array([])

    ctx_0, ctx_1 = _TASK2CONTEXTS[task][0]["context"], _TASK2CONTEXTS[task][1]["context"]
    env = _TASK2ENV[task]
    default_0, default_1 = env.get_default_context()[ctx_0], env.get_default_context()[ctx_1]
    current_metrics = defaultdict(list)
    metric_context_name = (
        _TASK2CONTEXTS[task][context_idx]["context"] if context_idx is not None else None
    )

    for seed_path in exp_path.iterdir():
        eval_file = seed_path / "eval.jsonl"
        if not eval_file.exists():
            continue
        with eval_file.open() as f:
            lines = [
                json.loads(line) for line in f if not json.loads(line)["aggregated_context_metric"]
            ]
        for line in lines:
            ctx_0_val, ctx_1_val = line["ctx"]["context"][ctx_0], line["ctx"]["context"][ctx_1]
            ret = float(line["return"])
            if train_setting == "double_box" or (
                metric_context_name in line["ctx"]["changed"] and len(line["ctx"]["changed"]) <= 1
            ):
                current_metrics[(ctx_0_val, ctx_1_val)].append(ret)

    agent_metrics = {k: np.array(v) for k, v in current_metrics.items()}
    if not agent_metrics:
        print(f"SKIPPING {seed_path} agent metrics empty. PLEASE DO NOT USE THESE PLOTS")
        return (np.array([]), np.array([]), np.array([]))

    normalized_score = match_and_normalize_metrics(
        agent_metrics, best_metrics[task], random_metrics[task]
    )

    interpolate_ranges = [
        [default_0 - 1e-6, default_0 + 1e-6],
        [default_1 - 1e-6, default_1 + 1e-6],
    ]
    if context_idx is not None:
        interpolate_ranges[context_idx] = _TASK2CONTEXTS[task][context_idx]["train_range"]
    else:
        interpolate_ranges[0] = _TASK2CONTEXTS[task][0]["train_range"]
        interpolate_ranges[1] = _TASK2CONTEXTS[task][1]["train_range"]

    interpolate_score, extrapolate_score, inter_extrapolate_score = [], [], []
    for ctx_key, score in normalized_score.items():
        in_range_0 = (
            ctx_key[0] >= interpolate_ranges[0][0] and ctx_key[0] <= interpolate_ranges[0][1]
        )
        in_range_1 = (
            ctx_key[1] >= interpolate_ranges[1][0] and ctx_key[1] <= interpolate_ranges[1][1]
        )
        if in_range_0 and in_range_1:
            interpolate_score.append(score)
        elif context_idx is not None or (not in_range_0 and not in_range_1):
            extrapolate_score.append(score)
        else:
            inter_extrapolate_score.append(score)

    return (
        np.array(interpolate_score),
        np.array(extrapolate_score),
        np.array(inter_extrapolate_score),
    )


def create_scores_from_df(df, type, metrics):
    aggregate_scores, aggregate_score_cis = {}, {}
    for algorithm in df.index:
        scores = []
        cis = []
        for metric in metrics:
            score_col = f"{type}_{metric}"
            ci_col = f"{type}_{metric}_ci"
            if score_col in df.columns and ci_col in df.columns:
                val = df.loc[algorithm, score_col]
                if pd.isna(val):
                    continue
                scores.append(val)
                ci = df.loc[algorithm, ci_col]
                if isinstance(ci, str):
                    ci = list(map(float, ci.strip("[]").split()))
                else:
                    ci = ci.tolist()
                cis.append(ci)
        if scores:
            aggregate_scores[algorithm] = np.array(scores)
            aggregate_score_cis[algorithm] = np.array(cis).T
    return aggregate_scores, aggregate_score_cis


def plot(df, metrics, target_dir="plots", task_name=None, modality=None):
    for met in metrics:
        types = ["Interp", "Extrap", "Inter_Extra"]
        aggregate_scores_type = {t: create_scores_from_df(df, t, [met])[0] for t in types}
        aggregate_score_cis_type = {t: create_scores_from_df(df, t, [met])[1] for t in types}

        aggregate_scores = {
            alg: np.concatenate(
                [aggregate_scores_type[t][alg] for t in types if alg in aggregate_scores_type[t]],
                axis=0,
            )
            for alg in df.index
        }
        aggregate_score_cis = {
            alg: np.concatenate(
                [
                    aggregate_score_cis_type[t][alg]
                    for t in types
                    if alg in aggregate_score_cis_type[t]
                ],
                axis=1,
            )
            for alg in df.index
        }

        available_metrics = [m for m in metrics if any(f"{t}_{m}" in df.columns for t in types)]
        title_fig = f"{task_name or 'Aggregated'} - {modality or 'Unknown'}"
        Path(target_dir).mkdir(exist_ok=True)

        m_names = [f"{TYPE_MAP[t]}" for t in types]
        fig, axes = plot_utils.plot_interval_estimates(
            aggregate_scores,
            aggregate_score_cis,
            metric_names=m_names,
            algorithms=list(aggregate_scores.keys()),
            colors=COLOR_MAP,
            row_height=0.5,
            subfigure_width=5.0,
            xlabel=title_fig,
            xlabel_y_coordinate=-0.3,
        )
        fig.subplots_adjust(left=0.3)
        plt.tight_layout()
        fig.savefig(f"{target_dir}/{title_fig.replace(' - ', '_')}_{met}.pdf")
        plt.close()


def plot_probability_of_improvement(
    df,
    reference_alg,
    types=["Interp", "Extrap", "Inter_Extra"],
    target_dir="plots",
    task_name=None,
    modality=None,
):
    algorithms = [
        alg
        for alg in df.index
        if alg != reference_alg
        and f"{types[0]}_IP" in df.columns
        and not pd.isna(df.loc[alg, f"{types[0]}_IP"])
    ]
    if not algorithms:
        return

    fig, axs = plt.subplots(1, len(types), figsize=(15, 3))
    if len(types) == 1:
        axs = [axs]

    for j, type in enumerate(types):
        probabilities = df.loc[algorithms, f"{type}_IP"].values
        probability_cis = df.loc[algorithms, f"{type}_IP_ci"].values
        cis_list = []
        for ci in probability_cis:
            if isinstance(ci, (list, np.ndarray)):
                cis_list.append([ci[0].item(), ci[1].item()])
            else:
                bounds = [float(x) for x in ci.strip("[]").split()]
                cis_list.append(bounds)
        cis_array = np.array(cis_list).T

        ax = axs[j]
        for i, alg in enumerate(algorithms):
            prob = probabilities[i]
            l, u = cis_array[:, i]
            color = COLOR_MAP[alg]
            ax.barh(y=i, width=u - l, left=l, height=0.6, color=color, alpha=0.75, label=alg)
            ax.vlines(x=prob, ymin=i - 0.3, ymax=i + 0.3, color="k", alpha=0.5)

        ax.set_yticks(range(len(algorithms)))
        if j == 0:
            ax.set_yticklabels(algorithms, fontsize="x-large")
        else:
            ax.set_yticklabels([])

        ax.set_title(f"{TYPE_MAP[type]}", fontsize="xx-large")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(True, axis="x", alpha=0.25)
        ax.axvline(x=0.5, color="r", linestyle="--")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))
        ax.tick_params(labelsize="x-large")

    title_fig = (
        f"{task_name or 'Aggregated'} - {modality or 'Unknown'} - Probability_of_Improvement"
    )
    fig.suptitle(f"P({reference_alg} $>$ Y)", fontsize="xx-large")
    plt.tight_layout()
    common_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    common_ax.set_xticks([])
    common_ax.set_yticks([])
    common_ax.set_xlabel("Probability", fontsize="xx-large")
    common_ax.xaxis.set_label_coords(0.5, -0.05)
    fig.savefig(f"{target_dir}/{title_fig.replace(' - ', '_')}.pdf", bbox_inches="tight")
    plt.close()


def _decorate_axis(ax, wrect=10, hrect=10, ticklabelsize='large'):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(length=0.1, width=0.1)
    ax.tick_params(axis='x', labelsize=ticklabelsize)
    ax.tick_params(axis='y', labelsize=ticklabelsize * 1.1)
    ax.spines['left'].set_position(('outward', hrect))
    ax.spines['bottom'].set_position(('outward', wrect))
    return ax

def plot_combined_aligned(df, reference_alg, target_dir="plots", task_name=None, modality=None):
    types = ["Interp", "Extrap", "Inter_Extra"]
    met = "IQM"
    algorithms = list(df.index)

    # Figure setup
    subfigure_width = 4.5
    row_height = 0.37
    figsize = (subfigure_width * 4, row_height * len(algorithms))
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    h = 0.5  # Slimmer bars as requested

    # Store scores for each type
    scores_per_type = {}

    # Plot IQM for each type
    for idx, type in enumerate(types):
        ax = axes[idx]
        aggregate_scores, aggregate_score_cis = create_scores_from_df(df, type, [met])
        # Store IQM scores for improvement calculation
        scores_per_type[type] = {alg: aggregate_scores[alg][0] for alg in algorithms if alg in aggregate_scores}
        for alg_idx, alg in enumerate(algorithms):
            if alg in aggregate_scores:
                score = aggregate_scores[alg][0]
                lower, upper = aggregate_score_cis[alg]
                lower = lower[0]
                upper = upper[0]
                ax.barh(
                    y=alg_idx,
                    width=upper - lower,
                    height=h,
                    left=lower,
                    color=COLOR_MAP[alg],
                    alpha=0.75
                )
                ax.vlines(
                    x=score,
                    ymin=alg_idx - (7.5 * h / 16),
                    ymax=alg_idx + (6 * h / 16),
                    color="k",
                    alpha=0.5
                )
        ax.set_title(f"{TYPE_MAP[type]}", fontsize=20)
        ax.set_xlabel("Normalized Score", fontsize=20)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_yticks(range(len(algorithms)))
        ax.set_yticklabels(algorithms, fontsize=34)
        _decorate_axis(ax, ticklabelsize=20, wrect=10, hrect=10)
        ax.spines['left'].set_visible(False)
        ax.grid(True, axis="x", alpha=0.25)

    # Plot Probability of Improvement
    ax = axes[3]
    type = "Extrap"
    plot_algorithms = [alg for alg in algorithms if alg != reference_alg]
    for alg in plot_algorithms:
        y_pos = algorithms.index(alg)  # Corrected y-position
        prob = df.loc[alg, f"{type}_IP"]
        ci = df.loc[alg, f"{type}_IP_ci"]
        if isinstance(ci, str):
            l, u = map(float, ci.strip("[]").split())
        else:
            l, u = ci
        l = l[0]
        u = u[0]
        ax.barh(
            y=y_pos,
            width=u - l,
            height=h,
            left=l,
            color=COLOR_MAP[alg],
            alpha=0.75
        )
        ax.vlines(
            x=prob,
            ymin=y_pos - (7.5 * h / 16),
            ymax=y_pos + (6 * h / 16),
            color="k",
            alpha=0.5
        )
    ax.set_title(f"P({reference_alg} $>$ Y) (Extrap)", fontsize=20)
    ax.set_xlabel("Probability", fontsize=20)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(True, axis="x", alpha=0.25)
    ax.axvline(x=0.5, color="r", linestyle="--")
    _decorate_axis(ax, ticklabelsize=20, wrect=10, hrect=10)
    ax.spines['left'].set_visible(False)

    # Compute improvements relative to alg_idx = 0
    improvements = {}
    alg0 = algorithms[0]
    for type in types:
        score_alg0 = scores_per_type[type][alg0]
        improvements[type] = {
            alg: {
                'score': scores_per_type[type][alg],
                'improvement': scores_per_type[type][alg] - score_alg0
            }
            for alg in algorithms
        }

    title_fig = f"{task_name or 'Aggregated'} - {modality or 'Unknown'} - Combined"
    plot_file = f"{target_dir}/{title_fig.replace(' - ', '_')}.pdf"
    txt_file = plot_file.replace('.pdf', '.txt')
    with open(txt_file, 'w') as f:
        for type in types:
            f.write(f"Type: {type}\n")
            for alg in algorithms:
                score = improvements[type][alg]['score']
                improvement = improvements[type][alg]['improvement']
                f.write(f"{alg}: {score:.4f} ({improvement:+.4f})\n")
            f.write("\n")

    # Adjust layout and save figure
    plt.subplots_adjust(left=0.2, wspace=0.22)
    Path(target_dir).mkdir(exist_ok=True)
    fig.savefig(plot_file, bbox_inches="tight")
    plt.tight_layout()
    plt.close()

    return fig, axes


def main(args):
    logdir = Path(args.exp_path)
    expert_metrics, random_metrics = get_expert_and_random_metrics(logdir)

    for task in args.tasks:
        if task == "dmc_walker":
            reference_alg = "DALI-S"
        else:
            reference_alg = r"DALI-S-$\rchi$"

        algorithms = [
            "no_ctx",
            "enc_dec_ctx",
            "pgm_ctx",
            "no_ctx_ctxtransformer",
            "no_ctx_ctxtransformer_coupling",
        ]
        for modality in args.modalities:
            df = compute_agg_all(
                logdir,
                [task],
                TRAIN_SETTINGS,
                modality,
                expert_metrics,
                random_metrics,
                algorithms,
                args.metrics,
                reference_alg
            )
            plot_combined_aligned(
                df,
                reference_alg,
                target_dir=args.output_dir,
                task_name=TASK_MAP[task],
                modality=MODALITY_MAP[modality],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument(
        "--exp_path",
        type=str,
        default="logs",
        help="Path to experiment logs.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=DEFAULT_TASKS, help="List of tasks to include."
    )
    parser.add_argument(
        "--modalities", nargs="+", default=DEFAULT_MODALITIES, help="List of modalities to include."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="List of metrics to plot (e.g., IQM, Optimality Gap).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots_rliable", help="Directory to save plots."
    )
    args = parser.parse_args()
    main(args)
