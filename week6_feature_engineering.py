import pandas as pd

# Load Dataset from week 5 
listings = pd.read_csv("cleaned_listing.csv")
sold = pd.read_csv("cleaned_sold.csv")

# Convert Data columns 
date_cols = ["CloseDate", "PurchaseContractDate", "ListingContractDate"]

for col in date_cols:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors="coerce")


# Feature Engineering 
# Price Ratio: measures how close the final sale price is to the original list price
sold["price_ratio"] = sold["ClosePrice"] / sold["OriginalListPrice"]

# Close-to-Original-List Ratio: same formula, kept as a separate metric for dashboard clarity
sold["close_to_original_list_ratio"] = sold["ClosePrice"] / sold["OriginalListPrice"]

# Price Per Square Foot: normalizes sale price by property size
sold["price_per_sqft"] = sold["ClosePrice"] / sold["LivingArea"]

# Days on Market: raw field showing how long the property was listed
sold["days_on_market"] = sold["DaysOnMarket"]

# Year, Month, and Year-Month features for time-series analysis
sold["year"] = sold["CloseDate"].dt.year
sold["month"] = sold["CloseDate"].dt.month
sold["yrmo"] = sold["CloseDate"].dt.to_period("M").astype(str)

# Listing-to-Contract Days: time from listing date to accepted offer date
sold["listing_to_contract_days"] = (
    sold["PurchaseContractDate"] - sold["ListingContractDate"]
).dt.days

# Contract-to-Close Days: time from accepted offer date to closing date
sold["contract_to_close_days"] = (
    sold["CloseDate"] - sold["PurchaseContractDate"]
).dt.days



# Sample output table
# This table demonstrates that the new columns were populated correctly.
sample_output = sold[
    [
        "ClosePrice",
        "OriginalListPrice",
        "LivingArea",
        "price_ratio",
        "close_to_original_list_ratio",
        "price_per_sqft",
        "days_on_market",
        "yrmo",
        "listing_to_contract_days",
        "contract_to_close_days",
    ]
].head(10)

print("\n===== Sample Output Table =====")
print(sample_output)


# Segmented summary by PropertyType
property_type_summary = sold.groupby("PropertyType").agg(
    avg_close_price=("ClosePrice", "mean"),
    avg_price_ratio=("price_ratio", "mean"),
    avg_close_to_original_list_ratio=("close_to_original_list_ratio", "mean"),
    avg_price_per_sqft=("price_per_sqft", "mean"),
    avg_days_on_market=("days_on_market", "mean"),
    avg_listing_to_contract_days=("listing_to_contract_days", "mean"),
    avg_contract_to_close_days=("contract_to_close_days", "mean"),
    count=("PropertyType", "size")
).reset_index()

print("\n===== Segment Summary by PropertyType =====")
print(property_type_summary)


# Segmented summary by CountyOrParish
county_summary = sold.groupby("CountyOrParish").agg(
    avg_close_price=("ClosePrice", "mean"),
    avg_price_ratio=("price_ratio", "mean"),
    avg_close_to_original_list_ratio=("close_to_original_list_ratio", "mean"),
    avg_price_per_sqft=("price_per_sqft", "mean"),
    avg_days_on_market=("days_on_market", "mean"),
    avg_listing_to_contract_days=("listing_to_contract_days", "mean"),
    avg_contract_to_close_days=("contract_to_close_days", "mean"),
    count=("CountyOrParish", "size")
).reset_index()

print("\n===== Segment Summary by CountyOrParish =====")
print(county_summary)

# Save outputs 
# sold.to_csv("week6_engineered_sold.csv", index=False)
# property_type_summary.to_csv("week6_property_type_summary.csv", index=False)
# county_summary.to_csv("week6_county_summary.csv", index=False)