import pandas as pd
from pathlib import Path


# ============================================================
# Week 7: Outlier Detection and Data Quality
# Applies IQR-based outlier flagging to both sold and listing datasets.
# Saves full flagged datasets and clean filtered datasets separately.
# ============================================================


OUTPUT_DIR = Path("week7_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


DATASETS = {
    "sold": "cleaned_sold.csv",
    "listing": "cleaned_listing.csv"
}


KEY_NUMERIC_FIELDS = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket"
]


BUSINESS_RULES = {
    "ClosePrice": lambda s: s.isna() | (s <= 0),
    "LivingArea": lambda s: s.isna() | (s <= 0),
    "DaysOnMarket": lambda s: s.isna() | (s < 0)
}


def apply_iqr_filtering(df, dataset_name):
    """
    Adds invalid-value flags and IQR outlier flags to key numeric columns.
    Returns:
        flagged_df: full dataset with flags
        clean_df: filtered dataset excluding invalid and outlier records
        summary_df: before/after comparison
    """

    flagged_df = df.copy()

    print(f"\n{'=' * 60}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'=' * 60}")
    print(f"Original shape: {flagged_df.shape}")

    available_fields = [
        col for col in KEY_NUMERIC_FIELDS
        if col in flagged_df.columns
    ]

    missing_fields = [
        col for col in KEY_NUMERIC_FIELDS
        if col not in flagged_df.columns
    ]

    if missing_fields:
        print(f"Warning: Missing columns in {dataset_name}: {missing_fields}")

    # Convert available numeric fields to numeric dtype.
    for col in available_fields:
        flagged_df[col] = pd.to_numeric(flagged_df[col], errors="coerce")

    invalid_flags = []
    outlier_flags = []

    # Apply business-rule invalid flags and IQR outlier flags.
    for col in available_fields:
        invalid_col = f"{col}_business_invalid_flag"
        outlier_col = f"{col}_iqr_outlier_flag"

        flagged_df[invalid_col] = BUSINESS_RULES[col](flagged_df[col])
        invalid_flags.append(invalid_col)

        valid_values = flagged_df.loc[~flagged_df[invalid_col], col].dropna()

        if valid_values.empty:
            print(f"\n{col}: No valid values available for IQR calculation.")
            flagged_df[f"{col}_iqr_lower_bound"] = pd.NA
            flagged_df[f"{col}_iqr_upper_bound"] = pd.NA
            flagged_df[outlier_col] = False
            outlier_flags.append(outlier_col)
            continue

        q1 = valid_values.quantile(0.25)
        q3 = valid_values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        flagged_df[f"{col}_iqr_lower_bound"] = lower
        flagged_df[f"{col}_iqr_upper_bound"] = upper

        flagged_df[outlier_col] = (
            (~flagged_df[invalid_col]) &
            (
                (flagged_df[col] < lower) |
                (flagged_df[col] > upper)
            )
        )

        outlier_flags.append(outlier_col)

        print(f"\n{col}")
        print(f"  Q1: {q1}")
        print(f"  Q3: {q3}")
        print(f"  IQR: {iqr}")
        print(f"  Lower bound: {lower}")
        print(f"  Upper bound: {upper}")
        print(f"  Business invalid count: {flagged_df[invalid_col].sum()}")
        print(f"  IQR outlier count: {flagged_df[outlier_col].sum()}")

    # Create overall flags.
    if invalid_flags:
        flagged_df["any_business_invalid_flag"] = flagged_df[invalid_flags].any(axis=1)
    else:
        flagged_df["any_business_invalid_flag"] = False

    if outlier_flags:
        flagged_df["any_iqr_outlier_flag"] = flagged_df[outlier_flags].any(axis=1)
    else:
        flagged_df["any_iqr_outlier_flag"] = False

    flagged_df["remove_from_clean_dataset"] = (
        flagged_df["any_business_invalid_flag"] |
        flagged_df["any_iqr_outlier_flag"]
    )

    clean_df = flagged_df.loc[~flagged_df["remove_from_clean_dataset"]].copy()

    # Build before/after summary.
    summary_rows = []

    summary_rows.append({
        "dataset": dataset_name,
        "metric": "row_count",
        "before_filtering": len(flagged_df),
        "after_filtering": len(clean_df),
        "change": len(clean_df) - len(flagged_df),
        "percent_change": (
            (len(clean_df) - len(flagged_df)) / len(flagged_df) * 100
            if len(flagged_df) > 0 else 0
        )
    })

    summary_rows.append({
        "dataset": dataset_name,
        "metric": "removed_records",
        "before_filtering": flagged_df["remove_from_clean_dataset"].sum(),
        "after_filtering": 0,
        "change": -flagged_df["remove_from_clean_dataset"].sum(),
        "percent_change": None
    })

    for col in available_fields:
        before_median = flagged_df[col].median()
        after_median = clean_df[col].median()

        summary_rows.append({
            "dataset": dataset_name,
            "metric": f"median_{col}",
            "before_filtering": before_median,
            "after_filtering": after_median,
            "change": after_median - before_median,
            "percent_change": (
                (after_median - before_median) / before_median * 100
                if pd.notna(before_median) and before_median != 0 else None
            )
        })

        summary_rows.append({
            "dataset": dataset_name,
            "metric": f"{col}_business_invalid_count",
            "before_filtering": flagged_df[f"{col}_business_invalid_flag"].sum(),
            "after_filtering": 0,
            "change": -flagged_df[f"{col}_business_invalid_flag"].sum(),
            "percent_change": None
        })

        summary_rows.append({
            "dataset": dataset_name,
            "metric": f"{col}_iqr_outlier_count",
            "before_filtering": flagged_df[f"{col}_iqr_outlier_flag"].sum(),
            "after_filtering": 0,
            "change": -flagged_df[f"{col}_iqr_outlier_flag"].sum(),
            "percent_change": None
        })

    summary_df = pd.DataFrame(summary_rows)

    print(f"\nFinal flagged shape: {flagged_df.shape}")
    print(f"Final clean filtered shape: {clean_df.shape}")

    return flagged_df, clean_df, summary_df


def main():
    all_summaries = []

    for dataset_name, input_file in DATASETS.items():
        input_path = Path(input_file)

        if not input_path.exists():
            print(f"\nSkipping {dataset_name}: file not found -> {input_file}")
            continue

        df = pd.read_csv(input_path, low_memory=False)

        flagged_df, clean_df, summary_df = apply_iqr_filtering(df, dataset_name)

        flagged_output = OUTPUT_DIR / f"week7_{dataset_name}_flagged_dataset.csv"
        clean_output = OUTPUT_DIR / f"week7_{dataset_name}_clean_filtered_dataset.csv"
        summary_output = OUTPUT_DIR / f"week7_{dataset_name}_filtering_summary.csv"

        flagged_df.to_csv(flagged_output, index=False)
        clean_df.to_csv(clean_output, index=False)
        summary_df.to_csv(summary_output, index=False)

        all_summaries.append(summary_df)

        print("\nSaved:")
        print(f"  {flagged_output}")
        print(f"  {clean_output}")
        print(f"  {summary_output}")

    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_summary_output = OUTPUT_DIR / "week7_combined_filtering_summary.csv"
        combined_summary.to_csv(combined_summary_output, index=False)

        print("\nCombined summary saved:")
        print(f"  {combined_summary_output}")

        print("\nCombined summary preview:")
        print(combined_summary)

    else:
        print("\nNo datasets were processed. Check your input file names.")


if __name__ == "__main__":
    main()