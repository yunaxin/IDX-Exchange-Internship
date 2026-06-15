import pandas as pd
from pathlib import Path
from datetime import date

# Week 1 - Monthly Dataset Aggregation

data_folder = Path("./data")

# Output file names 
listing_output = Path("CombinedListing_Residential.csv")
sold_output = Path("CombinedSold_Residential.csv")

def get_completed_months(start_year=2024, start_month=1):
    """
    Returns a list of YYYYMM strings from January 2024
    through the most recently completed calendar month.
    """
    today = date.today()

    if today.month == 1:
        end_year = today.year - 1
        end_month = 12
    else:
        end_year = today.year
        end_month = today.month - 1

    months = []
    year = start_year
    month = start_month

    while (year < end_year) or (year == end_year and month <= end_month):
        months.append(f"{year}{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1

    return months 

def load_monthly_files(prefix):
    """
    Loads monthly CSV files for either listings or sold. 
    """
    months = get_completed_months()
    dfs = []

    print(f"\nLoading files for {prefix}:")

    for yyyymm in months:
        file_path = data_folder / f"{prefix}{yyyymm}.csv"

        if file_path.exists():
            df = pd.read_csv(file_path, low_memory=False)

            #Row count before concatenation for each monthly file
            print(f"{file_path.name}: {len(df)} rows")

            dfs.append(df)
        else:
            print(f"Missing file: {file_path.name}")
    
    if not dfs:
        raise FileNotFoundError(f"No files found for prefix {prefix} in {data_folder}")

    return dfs

# ----------------
# LISTING DATASET
# ----------------

listing_dfs = load_monthly_files("CRMLSListing")

# Total row count before concatenation
listing_before_concat = sum(len(df) for df in listing_dfs)
print(f"\nListing total rows before concatenation: {listing_before_concat}")

# Concatenate all monthly listings files 
combined_listing = pd.concat(listing_dfs, ignore_index=True)

# Row count after concatenation
print(f"Listing rows after concatenation: {len(combined_listing)}")

# Row count before Residential filter
print(f"Listing rows before Residential filter: {len(combined_listing)}")

# Filter to PropertyType == 'Residential'
combined_listing_residential = combined_listing[
    combined_listing["PropertyType"].astype(str).str.strip() == "Residential"
].copy()

# Row count after Residential filter
print(f"Listing rows after Residential filter: {len(combined_listing_residential)}")


# ------------
# SOLD DATASET
# ------------

sold_dfs = load_monthly_files("CRMLSSold")

# Total row count before concatenation
sold_before_concat = sum(len(df) for df in sold_dfs)
print(f"\nSold total rows before concatenation: {sold_before_concat}")

# Concatenate all monthly sold files 
combined_sold = pd.concat(sold_dfs, ignore_index=True)

# Row count after concatenation
print(f"Sold rows after concatenation: {len(combined_sold)}")

# Row count before Residential filter
print(f"Sold rows before Residential filter: {len(combined_sold)}")

# Filter to PropertyType == 'Residential'
combined_sold_residential = combined_sold[
    combined_sold["PropertyType"].astype(str).str.strip() == "Residential"
].copy()

# Row count after Residential filter
print(f"Sold rows after Residential filter: {len(combined_sold_residential)}")

# -----------------
# SAVE OUTPUT FILES
# -----------------

combined_listing_residential.to_csv(listing_output, index=False)
combined_sold_residential.to_csv(sold_output, index=False)

print(f"\nSaved listing file: {listing_output}")
print(f"Saved sold file: {sold_output}")
