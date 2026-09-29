# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "hotel_bookings.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "jessemostipak/hotel-booking-demand",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:\n", df.head())

# import kagglehub
# import pandas as pd
# import os

# # Download the dataset
# path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")

# print("Dataset downloaded to:")
# print(path)

# # Find the CSV file
# csv_path = os.path.join(path, "hotel_bookings.csv")

# # Read the CSV file
# df = pd.read_csv(csv_path)

# # Display first 5 records
# print("\nFirst 5 records:")
# print(df.head())