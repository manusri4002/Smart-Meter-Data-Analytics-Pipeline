from src.ingestion import generate_raw_cer_dataset, load_raw_data

# Generate raw data and save to data/raw/cer_raw_data.csv
generate_raw_cer_dataset(
    output_filepath="data/raw/cer_raw_data.csv", num_meters=10, days=7
)

# Load it back to verify it works
raw_df = load_raw_data("data/raw/cer_raw_data.csv")

# Print out the first few rows
print("\nFirst 5 rows of raw data:")
print(raw_df.head())

print("\nDataset Info:")
print(raw_df.info())
