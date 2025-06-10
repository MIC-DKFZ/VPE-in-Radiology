import json
from pathlib import Path
import argparse
import pandas as pd


def combine_metrics_to_csv(base_dir: Path, output_csv: Path):
    # Define the output columns in the order you want them.
    header = [
        "dataset",
        "model",
        "ann_type",
        "text_prompt",
        "auroc",
    ]

    rows = []

    # Walk all paths under base_dir matching metrics.json
    for metrics_file in base_dir.rglob("metrics.json"):
        # Extract subdirectory structure (omitting the final "metrics.json")
        relative_parts = metrics_file.relative_to(base_dir).parts[:-1]

        # Safely parse each field, noting some may be missing
        dataset = relative_parts[0] if len(relative_parts) > 0 else None
        model = relative_parts[1] if len(relative_parts) > 1 else None
        ann_type = relative_parts[2] if len(relative_parts) > 2 else None
        text_prompt = len(relative_parts) > 5

        # Load JSON data
        with open(metrics_file, "r") as f:
            data = json.load(f)

        # Build the row for the CSV
        row = [
            dataset,
            model,
            ann_type,
            text_prompt,
            data["auroc"],
        ]
        rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    df.to_csv(output_csv, index=False)
    print(f"Combined metrics written to {output_csv}")

    # Define models and the desired row/column order
    models = ["BIOMEDCLIPMODEL", "BMC_CLIP_CF_MODEL"]
    datasets_order = [
        "PadChestGRTrain",
        "PadChestGRVal",
        "PadChestGRTest",
        "VinDrCXRTrain",
        "VinDrCXRTest",
        "NIH14",
        "JSRT",
    ]
    row_order = [
        ("None", False),
        ("crop", False),
        ("arrow", False),
        ("arrow", True),
        ("bbox", False),
        ("bbox", True),
        ("circle", False),
        ("circle", True),
    ]

    # Ensure text_prompt is boolean if it's not already
    df["text_prompt"] = df["text_prompt"].astype(bool)

    # Build tables for each model
    tables = {}
    for model in models:
        model_df = df[df["model"] == model]
        table = []
        for ann_type, text_prompt in row_order:
            row = []
            for dataset in datasets_order:
                match = model_df[
                    (model_df["ann_type"] == ann_type)
                    & (model_df["text_prompt"] == text_prompt)
                    & (model_df["dataset"] == dataset)
                ]
                val = match["auroc"].values[0] if not match.empty else None
                row.append(val)
            table.append(row)

        tables[model] = pd.DataFrame(
            table, columns=datasets_order, index=[f"{a},{b}" for a, b in row_order]
        )

    # Save or display the tables
    for model, table in tables.items():
        print(f"\n=== {model} ===")
        print(table.round(3).to_markdown())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine metrics from multiple experiments into a single metadata file."
    )
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        default=Path("experiments"),
        help="Path to the experiment directory containing metrics.json files.",
    )
    args = parser.parse_args()

    output_csv_path = args.experiment_dir / "comb_metrics.csv"
    combine_metrics_to_csv(args.experiment_dir, output_csv_path)
    print(f"Combined metrics CSV created at {output_csv_path}")
