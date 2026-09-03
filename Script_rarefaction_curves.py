# ============================================================
# Rarefaction Curve Analysis for Bracken Reports
# VS Code-friendly version with configuration section
# ============================================================

import os
import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit


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

# Sampling interval.
# Example:
#   10 = calculate every 10 reads
#   100 = calculate every 100 reads
DEPTH_STEP = 10

# Random seed for reproducible rarefaction results
RANDOM_SEED = 42

# Number of points used to draw the fitted logarithmic curve
SMOOTH_POINTS = 500

# Whether to shade the standard deviation around each fitted curve
SHOW_SD_SHADING = True

# Whether to display the interactive plot after saving it
SHOW_PLOT = True

# ------------------------------------------------------------
# Sample display names and colours
#
# Key   = sample_id (derived from the Bracken filename, i.e.
#         the filename with "_bracken.txt" removed)
# Value = dict with optional "name" and "color" keys
#
#   "name"  -> label shown in the plot legend
#              (falls back to the original sample_id if omitted)
#   "color" -> any Plotly/CSS colour string, e.g. "#1f77b4",
#              "rgb(31,119,180)", "steelblue"
#              (falls back to the default colour cycle if omitted)
#
# Example:
# SAMPLE_CONFIG = {
#     "sample1": {"name": "Control",   "color": "#1f77b4"},
#     "sample2": {"name": "Treatment", "color": "#d62728"},
# }
#
# Any sample not listed here uses its original sample_id as the
# name and the next colour from the default cycle.
# ------------------------------------------------------------
SAMPLE_CONFIG = {
    # "sample1": {"name": "Control",   "color": "#1f77b4"},
    # "sample2": {"name": "Treatment", "color": "#d62728"},
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

        # Remove "_bracken.txt" from filename
        sample_id = filename.replace("_bracken.txt", "")

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
# RAREFACTION
# ============================================================

def rarefaction_curve(data, sample_id, depth_step=10, random_seed=42):
    """
    Generate a rarefaction curve for a single sample.

    Parameters
    ----------
    data : pd.DataFrame
        Combined Bracken dataframe.

    sample_id : str
        Sample to analyze.

    depth_step : int
        Interval between sampling depths.

    random_seed : int
        Seed for reproducible results.

    Returns
    -------
    dict
        Dictionary containing sampling depth and number
        of observed taxa.
    """

    sample_data = data[
        data["sample_id"] == sample_id
    ].copy()

    if sample_data.empty:
        raise ValueError(
            f"No data found for sample: {sample_id}"
        )

    # Convert read counts to integers
    sample_data["new_est_reads"] = (
        sample_data["new_est_reads"]
        .round()
        .astype(int)
    )

    # Remove zero counts
    sample_data = sample_data[
        sample_data["new_est_reads"] > 0
    ]

    total_reads = sample_data["new_est_reads"].sum()

    if total_reads < depth_step:
        print(
            f"WARNING: Sample '{sample_id}' has only "
            f"{total_reads} reads."
        )
        return {}

    # --------------------------------------------------------
    # Create an individual read-level representation.
    #
    # Each taxonomy_id is repeated according to its read count.
    # This makes the rarefaction sampling biologically meaningful:
    # we sample individual reads rather than rows/taxa.
    # --------------------------------------------------------

    taxonomy_ids = np.repeat(
        sample_data["taxonomy_id"].values,
        sample_data["new_est_reads"].values
    )

    rng = np.random.default_rng(random_seed)

    rarefaction_data = {}

    # Include the maximum depth
    depths = list(
        range(
            depth_step,
            total_reads + 1,
            depth_step
        )
    )

    if total_reads not in depths:
        depths.append(total_reads)

    for depth in depths:

        # Randomly select reads without replacement
        sampled_reads = rng.choice(
            taxonomy_ids,
            size=depth,
            replace=False
        )

        unique_taxa = len(
            np.unique(sampled_reads)
        )

        rarefaction_data[depth] = unique_taxa

    return rarefaction_data


# ============================================================
# LOGARITHMIC MODEL
# ============================================================

def log_model(x, a, b):
    """
    Logarithmic model:

        y = a * log(x) + b
    """

    return a * np.log(x) + b


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
    Plot rarefaction curves with optional logarithmic
    standard-deviation shading.

    sample_config : dict, optional
        Maps sample_id -> {"name": ..., "color": ...}.
        Either key may be omitted; missing values fall back to
        the original sample_id (name) or the default colour
        cycle (color).
    """

    if sample_config is None:
        sample_config = {}

    fig = go.Figure()

    default_color_cycle = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf"
    ]

    color_idx = 0
    max_otus = 0

    for sample_id, data in rarefaction_results.items():

        if not data:
            print(
                f"Skipping '{sample_id}' because "
                f"no rarefaction data were generated."
            )
            continue

        config = sample_config.get(sample_id, {})

        sample_name = config.get("name", sample_id)

        depths = np.array(
            list(data.keys()),
            dtype=float
        )

        species_counts = np.array(
            list(data.values()),
            dtype=float
        )

        # ----------------------------------------------------
        # Fit logarithmic model
        # ----------------------------------------------------

        try:
            popt, _ = curve_fit(
                log_model,
                depths,
                species_counts,
                maxfev=10000
            )

            smooth_depths = np.linspace(
                depths.min(),
                depths.max(),
                SMOOTH_POINTS
            )

            smooth_counts = log_model(
                smooth_depths,
                *popt
            )

        except Exception as e:

            print(
                f"WARNING: Could not fit log model "
                f"for '{sample_name}': {e}"
            )

            smooth_depths = depths
            smooth_counts = species_counts

        # ----------------------------------------------------
        # Colour: use the user-defined colour if provided,
        # otherwise fall back to the next colour in the
        # default cycle.
        # ----------------------------------------------------

        color = config.get("color")

        if not color:
            color = default_color_cycle[
                color_idx % len(default_color_cycle)
            ]
            color_idx += 1

        # Convert colour to an RGBA fill for the SD shading.
        # Handles hex colours (#rrggbb); for named/rgb() colours
        # that Plotly understands natively, fall back to a
        # semi-transparent grey fill since we can't easily parse
        # arbitrary CSS colour strings ourselves.
        if color.startswith("#") and len(color) == 7:
            rgb = tuple(
                int(color[i:i + 2], 16)
                for i in (1, 3, 5)
            )
            fill_color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.20)"
        else:
            fill_color = "rgba(128, 128, 128, 0.20)"

        # ----------------------------------------------------
        # Main fitted curve
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=smooth_depths,
                y=smooth_counts,
                mode="lines",
                name=sample_name,
                line=dict(
                    color=color,
                    width=3
                )
            )
        )

        # ----------------------------------------------------
        # Original rarefaction points
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=depths,
                y=species_counts,
                mode="markers",
                name=f"{sample_name} observed",
                marker=dict(
                    color=color,
                    size=5,
                    opacity=0.2
                ),
                showlegend=False
            )
        )

        # ----------------------------------------------------
        # Standard deviation shading (optional)
        # ----------------------------------------------------

        if show_sd_shading:
            std_counts = np.std(species_counts) * 0.5

            upper = smooth_counts + std_counts
            lower = smooth_counts - std_counts

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([
                        smooth_depths,
                        smooth_depths[::-1]
                    ]),
                    y=np.concatenate([
                        upper,
                        lower[::-1]
                    ]),
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(
                        color="rgba(255,255,255,0)"
                    ),
                    name=f"SD: {sample_name}",
                    showlegend=False
                )
            )

        max_otus = max(
            max_otus,
            species_counts.max()
        )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(
        title=plot_title,
        xaxis_title="Number of Reads Sampled",
        yaxis_title="Unique OTUs (Taxa)",
        legend_title="Samples",
        template="plotly_white",
        hovermode="x unified",
        yaxis=dict(
            range=[0, max_otus * 1.2]
        )
    )

    # Make sure output directory exists
    output_directory = os.path.dirname(
        os.path.abspath(output_file)
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # Save HTML
    fig.write_html(output_file)

    print()
    print("=" * 60)
    print("Rarefaction analysis complete")
    print("=" * 60)
    print(f"Output file:")
    print(os.path.abspath(output_file))
    print("=" * 60)

    if SHOW_PLOT:
        fig.show()


# ============================================================
# SUCCESS MESSAGE
# ============================================================

def print_success_ascii():
    """
    Print a bunny with a SUCCESS message.
    """

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
    print("Bracken Rarefaction Curve Analysis")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input folder
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FOLDER):

        print()
        print("ERROR: Input folder does not exist.")
        print()
        print(f"Configured path:")
        print(f"  {INPUT_FOLDER}")
        print()
        print(
            "Please edit INPUT_FOLDER at the top "
            "of this script."
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Load Bracken data
    # --------------------------------------------------------

    print()
    print("Loading Bracken files...")
    print(f"Input folder: {INPUT_FOLDER}")

    data = load_bracken_files(
        INPUT_FOLDER
    )

    print()
    print(
        f"Loaded {data['sample_id'].nunique()} samples."
    )

    print(
        f"Loaded {len(data):,} taxa/sample records."
    )

    # --------------------------------------------------------
    # Generate rarefaction curves
    # --------------------------------------------------------

    rarefaction_results = {}

    print()
    print("Generating rarefaction curves...")
    print()

    for sample_id in data["sample_id"].unique():

        print(
            f"Processing: {sample_id}"
        )

        rarefaction_results[sample_id] = (
            rarefaction_curve(
                data,
                sample_id,
                depth_step=DEPTH_STEP,
                random_seed=RANDOM_SEED
            )
        )

    # --------------------------------------------------------
    # Plot results
    # --------------------------------------------------------

    plot_rarefaction_curves_html(
        rarefaction_results,
        OUTPUT_FILE,
        PLOT_TITLE,
        SAMPLE_CONFIG,
        show_sd_shading=SHOW_SD_SHADING
    )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print_success_ascii()