# ============================================================
# Rarefaction Curve Analysis for Bracken Reports
# Analytical (exact) version — VS Code-friendly, config at top
# ============================================================
#
# WHY THIS VERSION IS DIFFERENT FROM A "SIMULATE-and-plot" SCRIPT
# ------------------------------------------------------------
# A rarefaction curve is supposed to show the EXPECTED number of
# taxa observed when randomly subsampling m reads (without
# replacement) from a total of N reads. Simulating a single random
# draw at every depth and connecting the dots (as many quick
# scripts do) produces a noisy, seed-dependent curve that is not
# the expectation — and fitting a further curve (e.g. a*log(x)+b)
# to that noisy simulation compounds the problem, since a log
# curve has no asymptote and will keep "discovering" taxa forever
# under extrapolation, which is not how real accumulation curves
# behave.
#
# This script instead computes the EXACT expected value and exact
# variance of the rarefaction curve analytically, using the
# classic combinatorial (hypergeometric) formulas:
#
#   Expected richness at depth m (Hurlbert, 1971):
#       E[S(m)] = S - sum_i  C(N - n_i, m) / C(N, m)
#
#   Variance at depth m (Heck, van Belle & Simberloff, 1975):
#       Var[S(m)] = sum_i p_i(1-p_i)
#                   + 2 * sum_{i<j} ( p_ij - p_i*p_j )
#       where p_i  = C(N - n_i, m) / C(N, m)   (prob. taxon i absent)
#             p_ij = C(N - n_i - n_j, m) / C(N, m) (prob. both absent)
#
# N = total reads in the sample, n_i = reads assigned to taxon i,
# S = number of taxa, C(.,.) = binomial coefficient.
#
# This is exactly what vegan::rarefy() / QIIME2 / PAST compute.
# It is deterministic (no RANDOM_SEED needed), reproducible, and
# does not require building huge per-read arrays or repeated
# resampling.
# ============================================================

import os
import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.special import gammaln


# ============================================================
# CONFIGURATION
# ============================================================
# Edit ONLY this section before running the script.

# Folder containing your Bracken .txt files
INPUT_FOLDER = "/home/viroicbas2023/Documents/Ricardo/Grutas16S/GrackenCaverna16S/Output/bracken"

# Output HTML file
OUTPUT_FILE = "/home/viroicbas2023/Documents/Ricardo/Grutas16S/Rarefaction_curves_results/rarefaction_curves.html"

# Plot title
PLOT_TITLE = "Rarefaction Curve"

# Minimum spacing (in reads) between evaluated depths.
# The exact formula is smooth, so you don't need a fine step for
# a clean-looking curve — this mainly matters for very small
# samples. For large samples, MAX_DEPTH_POINTS below caps how
# many depths are actually evaluated (for speed), regardless of
# how small DEPTH_STEP is.
DEPTH_STEP = 10

# Hard cap on the number of depth points evaluated per sample.
# The variance calculation is O(S^2) per depth (S = number of
# taxa in that sample), so this keeps runtime bounded even for
# samples with millions of reads. 200 points is more than enough
# to draw a smooth curve, since the underlying formula has no
# sampling noise to average out.
MAX_DEPTH_POINTS = 200

# Above this number of taxa in a sample, the exact pairwise
# covariance term in the variance formula (O(S^2) per depth) is
# skipped for speed, and only the per-taxon term is used. This
# is an UNDERESTIMATE-leaning approximation in most cases, since
# the omitted covariance terms are typically negative for
# without-replacement sampling — but it keeps very taxon-rich
# samples tractable. A console warning is printed when this
# happens.
MAX_TAXA_FOR_EXACT_VARIANCE = 800

# Whether to shade the (exact) standard-deviation band around
# each curve.
SHOW_SD_SHADING = True

# Whether to display the interactive plot after saving it
SHOW_PLOT = True

# ------------------------------------------------------------
# Sample display names and colours
#
# Key   = sample_id (derived from the Bracken filename, i.e.
#         the filename with "_breport.txt" removed)
# Value = dict with optional "name" and "color" keys
#
#   "name"  -> label shown in the plot legend
#              (falls back to the original sample_id if omitted)
#   "color" -> any Plotly/CSS colour string, e.g. "#1f77b4",
#              "rgb(31,119,180)", "steelblue"
#              (falls back to the default colour cycle if omitted)
#
# Any sample not listed here uses its original sample_id as the
# name and the next colour from the default cycle.
# ------------------------------------------------------------
SAMPLE_CONFIG = {
    "1": {"name": "T-I",  "color": "#1f77b4"},
    "2": {"name": "T-I",  "color": "#ff7f0e"},
    "3": {"name": "M-II", "color": "#2ca02c"},
    "4": {"name": "A-I",  "color": "#d62728"},
    "5": {"name": "MV-I", "color": "#9467bd"},
    "6": {"name": "M-I",  "color": "#8c564b"},
    "7": {"name": "MN-I", "color": "#e377c2"},
}


# ============================================================
# LOAD BRACKEN FILES
# ============================================================

def load_bracken_files(input_folder):
    """
    Load Bracken report files from a specified folder.

    Expected columns:
        taxonomy_id
        new_est_reads

    Returns:
        pd.DataFrame
    """

    file_paths = glob.glob(os.path.join(input_folder, "*.txt"))

    if not file_paths:
        raise FileNotFoundError(
            f"No .txt files were found in:\n{input_folder}"
        )

    dataframes = []

    for file in file_paths:

        filename = os.path.basename(file)

        # Remove "_breport.txt" from filename
        # (.strip() guards against stray spaces in filenames,
        #  e.g. "1 _breport.txt" -> "1" instead of "1 ")
        sample_id = filename.replace("_breport.txt", "").strip()

        try:
            df = pd.read_csv(
                file,
                sep="\t",
                usecols=["taxonomy_id", "new_est_reads"]
            )
        except ValueError as e:
            print(f"\nWARNING: Could not read required columns from:")
            print(f"  {file}")
            print(f"  {e}")
            continue

        df["sample_id"] = sample_id

        dataframes.append(df)

    if not dataframes:
        raise ValueError(
            "No valid Bracken files could be loaded."
        )

    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    # Make sure read counts are numeric
    combined_df["new_est_reads"] = pd.to_numeric(
        combined_df["new_est_reads"],
        errors="coerce"
    )

    combined_df = combined_df.dropna(
        subset=["taxonomy_id", "new_est_reads"]
    )

    # Remove zero/negative read counts
    combined_df = combined_df[
        combined_df["new_est_reads"] > 0
    ]

    return combined_df


# ============================================================
# EXACT COMBINATORICS (log-space, numerically stable)
# ============================================================

def _log_choose(n, k):
    """
    log( C(n, k) ), vectorised, numerically stable via gammaln.

    Returns -inf wherever the binomial coefficient is not defined
    or is zero (k < 0, k > n, or n < 0), so downstream exp() gives
    a clean 0 instead of nan/inf.
    """

    n = np.asarray(n, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    n_b, k_b = np.broadcast_arrays(n, k)

    valid = (k_b >= 0) & (k_b <= n_b) & (n_b >= 0)

    # Compute everywhere (safe placeholder values where invalid,
    # to avoid warnings), then mask.
    n_safe = np.where(valid, n_b, 0.0)
    k_safe = np.where(valid, k_b, 0.0)

    log_val = (
        gammaln(n_safe + 1.0)
        - gammaln(k_safe + 1.0)
        - gammaln(n_safe - k_safe + 1.0)
    )

    return np.where(valid, log_val, -np.inf)


def _choose_depths(total_reads, depth_step, max_points):
    """
    Choose an evenly-spaced (and capped) set of subsample depths
    to evaluate, always including the full total_reads as the
    final point.
    """

    depths = np.arange(depth_step, total_reads + 1, depth_step, dtype=np.int64)

    if depths.size == 0 or depths[-1] != total_reads:
        depths = np.append(depths, total_reads)

    if depths.size > max_points:
        idx = np.linspace(0, depths.size - 1, max_points)
        idx = np.unique(np.round(idx).astype(int))
        depths = depths[idx]
        if depths[-1] != total_reads:
            depths = np.append(depths, total_reads)

    return depths


# ============================================================
# ANALYTICAL RAREFACTION (Hurlbert 1971 / Heck et al. 1975)
# ============================================================

def rarefaction_curve_analytical(
    data,
    sample_id,
    depth_step=10,
    max_depth_points=200,
    max_taxa_for_exact_variance=800,
    compute_variance=True,
):
    """
    Compute the EXACT expected rarefaction curve (and, optionally,
    its exact variance) for a single sample.

    Parameters
    ----------
    data : pd.DataFrame
        Combined Bracken dataframe (needs 'sample_id',
        'taxonomy_id', 'new_est_reads').
    sample_id : str
        Sample to analyze.
    depth_step : int
        Minimum spacing between evaluated depths.
    max_depth_points : int
        Hard cap on number of depths evaluated (for speed).
    max_taxa_for_exact_variance : int
        Above this many taxa, skip the O(S^2) pairwise covariance
        term in the variance and warn.
    compute_variance : bool
        Whether to compute variance/SD at all.

    Returns
    -------
    dict with keys:
        "depths"       : np.ndarray of subsample sizes
        "expected_S"   : np.ndarray, exact expected richness
        "sd"           : np.ndarray or None, exact SD of richness
        "variance_exact": bool, whether covariance term was included
    """

    sample_data = data[data["sample_id"] == sample_id]

    counts = (
        sample_data["new_est_reads"]
        .round()
        .astype(np.int64)
        .values
    )
    counts = counts[counts > 0]

    S = counts.size
    N = int(counts.sum()) if S > 0 else 0

    if S == 0 or N < depth_step:
        print(
            f"WARNING: Sample '{sample_id}' has only "
            f"{N} reads across {S} taxa — skipping."
        )
        return {}

    depths = _choose_depths(N, depth_step, max_depth_points)

    # ----------------------------------------------------------
    # Expected richness: E[S(m)] = S - sum_i C(N-n_i, m)/C(N, m)
    # ----------------------------------------------------------

    log_C_N_m = _log_choose(N, depths)                     # (D,)
    N_minus_ni = (N - counts)[:, None]                      # (S,1)
    log_C_Nni_m = _log_choose(N_minus_ni, depths[None, :])  # (S,D)

    p_i = np.exp(log_C_Nni_m - log_C_N_m[None, :])          # (S,D)
    p_i = np.clip(p_i, 0.0, 1.0)

    expected_S = S - p_i.sum(axis=0)

    # ----------------------------------------------------------
    # Variance (Heck, van Belle & Simberloff, 1975)
    # ----------------------------------------------------------

    sd = None
    variance_exact = False

    if compute_variance:
        var_term1 = (p_i * (1.0 - p_i)).sum(axis=0)  # (D,)

        if S <= max_taxa_for_exact_variance:
            D = depths.size
            variance = np.empty(D, dtype=np.float64)

            ni = counts[:, None]
            nj = counts[None, :]
            N_minus_ni_nj = N - ni - nj  # (S,S), reused across depths

            for d_idx in range(D):
                m = depths[d_idx]

                log_Cij = _log_choose(N_minus_ni_nj, m)          # (S,S)
                log_pij = log_Cij - log_C_N_m[d_idx]
                pij = np.exp(log_pij)
                pij = np.clip(pij, 0.0, 1.0)
                np.fill_diagonal(pij, 0.0)

                pi_d = p_i[:, d_idx]
                cross = pij - np.outer(pi_d, pi_d)
                np.fill_diagonal(cross, 0.0)

                # sum_{i<j} = half the full (symmetric, zero-diag) sum
                pairwise_sum = cross.sum() / 2.0

                variance[d_idx] = var_term1[d_idx] + 2.0 * pairwise_sum

            variance = np.clip(variance, 0.0, None)
            variance_exact = True

        else:
            print(
                f"NOTE: Sample '{sample_id}' has {S} taxa "
                f"(> {max_taxa_for_exact_variance}); using the "
                f"per-taxon variance term only (covariance term "
                f"skipped for speed). This tends to be a slight "
                f"over-estimate of the true variance."
            )
            variance = np.clip(var_term1, 0.0, None)
            variance_exact = False

        sd = np.sqrt(variance)

    return {
        "depths": depths,
        "expected_S": expected_S,
        "sd": sd,
        "variance_exact": variance_exact,
    }


# ============================================================
# PLOT RAREFACTION CURVES
# ============================================================

def plot_rarefaction_curves_html(
    rarefaction_results,
    output_file,
    plot_title,
    sample_config=None,
    show_sd_shading=True
):
    """
    Plot exact analytical rarefaction curves, each with an
    optional +/- 1 SD shaded band (SD computed exactly, per
    depth — not a fitted or ad-hoc value).
    """

    if sample_config is None:
        sample_config = {}

    fig = go.Figure()

    default_color_cycle = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    color_idx = 0
    max_otus = 0.0

    for sample_id, result in rarefaction_results.items():

        if not result:
            print(
                f"Skipping '{sample_id}' because "
                f"no rarefaction data were generated."
            )
            continue

        config = sample_config.get(sample_id, {})
        sample_name = config.get("name", sample_id)

        depths = result["depths"].astype(float)
        expected_S = result["expected_S"]
        sd = result["sd"]

        color = config.get("color")
        if not color:
            color = default_color_cycle[color_idx % len(default_color_cycle)]
            color_idx += 1

        if color.startswith("#") and len(color) == 7:
            rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
            fill_color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.20)"
        else:
            fill_color = "rgba(128, 128, 128, 0.20)"

        # ----------------------------------------------------
        # Exact expected curve
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=depths,
                y=expected_S,
                mode="lines",
                name=sample_name,
                line=dict(color=color, width=3)
            )
        )

        # ----------------------------------------------------
        # Exact SD shading (optional)
        # ----------------------------------------------------

        if show_sd_shading and sd is not None:

            upper = expected_S + sd
            lower = np.clip(expected_S - sd, 0, None)

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([depths, depths[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"SD: {sample_name}",
                    showlegend=False
                )
            )

            max_otus = max(max_otus, upper.max())
        else:
            max_otus = max(max_otus, expected_S.max())

    fig.update_layout(
        title=plot_title,
        xaxis_title="Number of Reads Sampled",
        yaxis_title="Expected Unique OTUs (Taxa)",
        legend_title="Samples",
        template="plotly_white",
        hovermode="x unified",
        yaxis=dict(range=[0, max_otus * 1.2 if max_otus > 0 else 1])
    )

    output_directory = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_directory, exist_ok=True)

    fig.write_html(output_file)

    print()
    print("=" * 60)
    print("Rarefaction analysis complete")
    print("=" * 60)
    print("Output file:")
    print(os.path.abspath(output_file))
    print("=" * 60)

    if SHOW_PLOT:
        fig.show()


# ============================================================
# SUCCESS MESSAGE
# ============================================================

def print_success_ascii():
    print(
        r"""
  SUCCESS !!

   __     __
  /\_/|   |\/_\
   |U|___|U|
   |       |
   | ,   , |
  (  = Y =  )
   |      |
  /|       |\
  \| |   | |/
 (_|_|___|_|_)

------------------------------------------------
        """
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Bracken Rarefaction Curve Analysis (exact analytical method)")
    print("=" * 60)

    if not os.path.exists(INPUT_FOLDER):
        print()
        print("ERROR: Input folder does not exist.")
        print()
        print("Configured path:")
        print(f"  {INPUT_FOLDER}")
        print()
        print("Please edit INPUT_FOLDER at the top of this script.")
        raise SystemExit(1)

    print()
    print("Loading Bracken files...")
    print(f"Input folder: {INPUT_FOLDER}")

    data = load_bracken_files(INPUT_FOLDER)

    print()
    print(f"Loaded {data['sample_id'].nunique()} samples.")
    print(f"Loaded {len(data):,} taxa/sample records.")

    rarefaction_results = {}

    print()
    print("Computing exact rarefaction curves...")
    print()

    for sample_id in data["sample_id"].unique():
        print(f"Processing: {sample_id}")

        rarefaction_results[sample_id] = rarefaction_curve_analytical(
            data,
            sample_id,
            depth_step=DEPTH_STEP,
            max_depth_points=MAX_DEPTH_POINTS,
            max_taxa_for_exact_variance=MAX_TAXA_FOR_EXACT_VARIANCE,
            compute_variance=SHOW_SD_SHADING,
        )

    plot_rarefaction_curves_html(
        rarefaction_results,
        OUTPUT_FILE,
        PLOT_TITLE,
        SAMPLE_CONFIG,
        show_sd_shading=SHOW_SD_SHADING
    )

    print_success_ascii()