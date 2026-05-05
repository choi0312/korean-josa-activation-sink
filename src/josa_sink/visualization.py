from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def savefig(path: str, dpi: int = 160):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, bbox_inches="tight", dpi=dpi)

def plot_layer_diff_curve(stats_df: pd.DataFrame, title: str, output_path: str, y_col: str = "mean_diff_josa_minus_nonjosa", dpi: int = 160):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(stats_df["layer"], stats_df[y_col], marker="o", linewidth=1.5)
    ax.axhline(0, linestyle="--", linewidth=1)
    if "significant_fdr_0_05" in stats_df.columns:
        sig = stats_df["significant_fdr_0_05"].fillna(False)
        ax.scatter(stats_df.loc[sig, "layer"], stats_df.loc[sig, y_col], marker="*", s=90, label="FDR<0.05")
    ax.set_xlabel("layer")
    ax.set_ylabel("JOSA - NON-JOSA")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    savefig(output_path, dpi=dpi)
    plt.close(fig)

def plot_last_layer_box(df: pd.DataFrame, metric: str, output_path: str, title: str, dpi: int = 160):
    last_layer = int(df["layer"].max())
    work = df[(df["layer"] == last_layer) & (~df["is_special"])].copy()
    order = ["JOSA", "NOUN", "VERB_OR_ADJ", "MODIFIER", "AUX_OR_SUFFIX", "OTHER_LEXICAL", "OTHER", "PUNCT_OR_SYMBOL"]
    labels = [x for x in order if (work["coarse_pos"] == x).any()]
    data = [work.loc[work["coarse_pos"] == lab, metric].dropna().values for lab in labels]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(title + f" | last layer={last_layer}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    savefig(output_path, dpi=dpi)
    plt.close(fig)

def plot_fakeq_summary(fakeq_summary: pd.DataFrame, value_col: str, output_path: str, title: str, dpi: int = 160):
    conditions_order = ["all_non_special", "matched_josa_only", "matched_nonjosa_only", "matched_random_non_special"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for cond in conditions_order:
        sub = fakeq_summary[fakeq_summary["condition"] == cond].sort_values("layer")
        if len(sub) == 0:
            continue
        sem_col = "sem_delta_loss" if value_col == "mean_delta_loss" else "sem_delta_loss_per_token"
        ax.plot(sub["layer"], sub[value_col], marker="o", label=cond)
        if sem_col in sub.columns:
            ax.fill_between(sub["layer"], sub[value_col] - 1.96 * sub[sem_col].fillna(0), sub[value_col] + 1.96 * sub[sem_col].fillna(0), alpha=0.15)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("hooked decoder layer")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    savefig(output_path, dpi=dpi)
    plt.close(fig)
