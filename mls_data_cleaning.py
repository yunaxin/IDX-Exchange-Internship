import pandas as pd
import numpy as np

# Load datasets
sold = pd.read_csv("CombinedSold_Residential.csv", low_memory=False)
listing = pd.read_csv("CombinedListing_Residential.csv", low_memory=False)

print("Before cleaning:")
print(f"Sold rows: {sold.shape[0]}, columns: {sold.shape[1]}")
print(f"Listing rows: {listing.shape[0]}, columns: {listing.shape[1]}")


# Helper cleaning function
def clean_mls_data(df, dataset_name):
    df = df.copy()

    print(f"\n===== Cleaning {dataset_name} dataset =====")

    before_rows = df.shape[0]

    # Convert date columns
    date_cols = [
        "CloseDate",
        "PurchaseContractDate",
        "ListingContractDate",
        "ContractStatusChangeDate"
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert numeric columns
    numeric_cols = [
        "ClosePrice",
        "ListPrice",
        "OriginalListPrice",
        "LivingArea",
        "LotSizeAcres",
        "BedroomsTotal",
        "BathroomsTotalInteger",
        "DaysOnMarket",
        "Latitude",
        "Longitude"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle missing values
    # Drop rows missing critical fields
    critical_cols = ["ClosePrice", "LivingArea"]
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])

    # Date consistency flags
    if "ListingContractDate" in df.columns and "CloseDate" in df.columns:
        df["listing_after_close_flag"] = df["ListingContractDate"] > df["CloseDate"]
    else:
        df["listing_after_close_flag"] = False

    if "PurchaseContractDate" in df.columns and "CloseDate" in df.columns:
        df["purchase_after_close_flag"] = df["PurchaseContractDate"] > df["CloseDate"]
    else:
        df["purchase_after_close_flag"] = False

    if "ListingContractDate" in df.columns and "PurchaseContractDate" in df.columns:
        df["negative_timeline_flag"] = df["ListingContractDate"] > df["PurchaseContractDate"]
    else:
        df["negative_timeline_flag"] = False


    # Geographic data flags
    if "Latitude" in df.columns and "Longitude" in df.columns:
        df["missing_coordinates_flag"] = df["Latitude"].isna() | df["Longitude"].isna()
        df["zero_coordinates_flag"] = (df["Latitude"] == 0) | (df["Longitude"] == 0)
        df["positive_longitude_flag"] = df["Longitude"] > 0

        # California rough coordinate range
        df["implausible_coordinates_flag"] = (
            (df["Latitude"] < 32) |
            (df["Latitude"] > 42) |
            (df["Longitude"] < -125) |
            (df["Longitude"] > -114)
        )
    else:
        df["missing_coordinates_flag"] = False
        df["zero_coordinates_flag"] = False
        df["positive_longitude_flag"] = False
        df["implausible_coordinates_flag"] = False

    # Invalid numeric value flags
    if "ClosePrice" in df.columns:
        df["invalid_close_price_flag"] = df["ClosePrice"] <= 0
    else:
        df["invalid_close_price_flag"] = False

    if "LivingArea" in df.columns:
        df["invalid_living_area_flag"] = df["LivingArea"] <= 0
    else:
        df["invalid_living_area_flag"] = False

    if "DaysOnMarket" in df.columns:
        df["invalid_days_on_market_flag"] = df["DaysOnMarket"] < 0
    else:
        df["invalid_days_on_market_flag"] = False

    if "BedroomsTotal" in df.columns:
        df["invalid_bedrooms_flag"] = df["BedroomsTotal"] < 0
    else:
        df["invalid_bedrooms_flag"] = False

    if "BathroomsTotalInteger" in df.columns:
        df["invalid_bathrooms_flag"] = df["BathroomsTotalInteger"] < 0
    else:
        df["invalid_bathrooms_flag"] = False

    # Remove unnecessary columns
    # Only removes if they exist
    columns_to_drop = [
        "Unnamed: 0",
        "ModificationTimestamp",
        "PhotosChangeTimestamp",
        "OriginatingSystemID"
    ]

    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Print validation summary
    print(f"Before rows: {before_rows}")
    print(f"After rows: {df.shape[0]}")
    print(f"Columns after cleaning: {df.shape[1]}")

    print("\nData types:")
    print(df.dtypes)

    print("\nDate consistency flag counts:")
    print("listing_after_close_flag:", df["listing_after_close_flag"].sum())
    print("purchase_after_close_flag:", df["purchase_after_close_flag"].sum())
    print("negative_timeline_flag:", df["negative_timeline_flag"].sum())

    print("\nGeographic data quality summary:")
    print("missing_coordinates_flag:", df["missing_coordinates_flag"].sum())
    print("zero_coordinates_flag:", df["zero_coordinates_flag"].sum())
    print("positive_longitude_flag:", df["positive_longitude_flag"].sum())
    print("implausible_coordinates_flag:", df["implausible_coordinates_flag"].sum())

    print("\nInvalid numeric value counts:")
    print("invalid_close_price_flag:", df["invalid_close_price_flag"].sum())
    print("invalid_living_area_flag:", df["invalid_living_area_flag"].sum())
    print("invalid_days_on_market_flag:", df["invalid_days_on_market_flag"].sum())
    print("invalid_bedrooms_flag:", df["invalid_bedrooms_flag"].sum())
    print("invalid_bathrooms_flag:", df["invalid_bathrooms_flag"].sum())

    return df



# Clean both datasets
cleaned_sold = clean_mls_data(sold, "sold")
cleaned_listing = clean_mls_data(listing, "listing")


# Save cleaned datasets
cleaned_sold.to_csv("cleaned_sold.csv", index=False)
cleaned_listing.to_csv("cleaned_listing.csv", index=False)

print("\nCleaning complete!")
print("Saved: cleaned_sold.csv")
print("Saved: cleaned_listing.csv")