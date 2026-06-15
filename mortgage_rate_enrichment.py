import pandas as pd 

# Load the datasets 
sold = pd.read_csv("CombinedSold_Residential.csv", low_memory=False)
listing = pd.read_csv("CombinedListing_Residential.csv", low_memory=False)

# Fetch the mortgage rate data from FRED
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

mortgage = pd.read_csv(url)
mortgage.columns = ['date', 'rate_30yr_fixed']
mortgage['date'] = pd.to_datetime(mortgage['date'])

# Resample weekly rates to monthly averages
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (
mortgage.groupby('year_month')['rate_30yr_fixed']
.mean().reset_index()
)

# Create a matching year_monthly key on the MLS datasets 
# Sold Dataset - key off CloseDate
sold['year_month'] = pd.to_datetime(sold['CloseDate']).dt.to_period('M')

# Listings dataset — key off ListingContractDate
listing['year_month'] = pd.to_datetime(
listing['ListingContractDate']).dt.to_period('M')


# Merge
sold_with_rates = sold.merge(mortgage_monthly, on='year_month', how='left')
listing_with_rates = listing.merge(mortgage_monthly, on='year_month', how='left')


# Validate
# Check for any unmatched rows (rate should not be null)
print(sold_with_rates['rate_30yr_fixed'].isnull().sum())
print(listing_with_rates['rate_30yr_fixed'].isnull().sum())

# Preview
print(sold_with_rates[['CloseDate', 'year_month', 'ClosePrice',
'rate_30yr_fixed']].head())

# Save 
sold_with_rates.to_csv("sold_with_rates.csv", index=False)
listing_with_rates.to_csv("listings_with_rates.csv", index=False)
