# ============================================================
# 1. IMPORT LIBRARIES AND LOAD DATASET
# ============================================================

# Install dependencies if required:
# pip install kagglehub[pandas-datasets]

import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from kagglehub import KaggleDatasetAdapter


# Define the dataset file to load
file_path = "hotel_bookings.csv"

# Load the latest version of the Kaggle dataset
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "jessemostipak/hotel-booking-demand",
    file_path
)


# ============================================================
# 2. INITIAL DATA INSPECTION
# ============================================================

# Display the first five records
print("First 5 records:\n", df.head())

# Display the number of observations and attributes
print("Total observations:", df.shape[0])
print("Total attributes:", df.shape[1])

# Display the complete dataset shape
print("Dataset shape:", df.shape)


# ============================================================
# 3. TARGET VARIABLE AND CLASS DISTRIBUTION
# ============================================================

# Define the target variable
target = "is_canceled"

print("\nTarget Variable:", target)

# Display the unique target classes
print("Unique Classes:", sorted(df[target].unique()))

# Display the number of target classes
print("Number of Classes:", df[target].nunique())

# Calculate class counts and percentages
counts = df[target].value_counts().sort_index()
percentages = df[target].value_counts(normalize=True).sort_index() * 100

# Print class distribution in a formatted table
print("\nClass Distribution:")
print(f"{'Class':<20} {'Count':>12} {'Percentage':>12}")
print("-" * 46)
print(f"{'Not cancelled':<20} {counts[0]:>12,} {percentages[0]:>11.2f}%")
print(f"{'Cancelled':<20} {counts[1]:>12,} {percentages[1]:>11.2f}%")


# Plot the target class distribution
fig, ax = plt.subplots(figsize=(7, 5))

labels = ["Not Cancelled (0)", "Cancelled (1)"]
colors = ["#1f77b4", "#ff7f0e"]

bars = ax.bar(
    labels,
    counts.values,
    color=colors,
    width=0.45
)

# Add count and percentage labels above each bar
for bar, count, pct in zip(bars, counts.values, percentages.values):
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1200,
        f"{count:,}\n({pct:.2f}%)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

ax.set_ylabel("Frequency", fontsize=11)
ax.set_title(
    "Target Class Distribution (is_canceled)",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylim(0, max(counts.values) * 1.18)

# Remove unnecessary chart borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()


# ============================================================
# 4. MISSING VALUE ANALYSIS
# ============================================================

# Count missing values for each column
missing_values = df.isnull().sum()

# Keep only columns containing missing values
missing_values = (
    missing_values[missing_values > 0]
    .sort_values(ascending=False)
)

# Calculate the percentage of missing values
missing_percentage = (missing_values / len(df)) * 100

# Create a summary table
missing_summary = pd.DataFrame({
    "Missing Values": missing_values,
    "Percentage": missing_percentage.round(2)
})

print("\nMissing Values:")
print(missing_summary)

# Plot missing values by column
missing_values.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Missing Values by Column")
plt.xlabel("Columns")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 5. DUPLICATE RECORD ANALYSIS
# ============================================================

# Count exact duplicate records
duplicate_count = df.duplicated().sum()

# Calculate the percentage of duplicate records
duplicate_percentage = (
    duplicate_count / len(df)
) * 100

print("\nDuplicate Record Analysis:")
print(f"Total records: {len(df):,}")
print(f"Duplicate records: {duplicate_count:,}")
print(f"Duplicate percentage: {duplicate_percentage:.2f}%")

# Calculate the percentage of non-duplicate records
non_duplicate_percentage = (
    (len(df) - duplicate_count) / len(df)
) * 100

# Store the percentages for plotting
duplicate_percentage_data = pd.Series({
    "Non-Duplicate": non_duplicate_percentage,
    "Duplicate": duplicate_percentage
})

# Plot duplicate and non-duplicate percentages
plt.figure(figsize=(8, 5))

duplicate_percentage_data.plot(kind="bar")

plt.title("Percentage of Duplicate and Non-Duplicate Records")
plt.xlabel("Record Type")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)

# Add percentage labels
for i, value in enumerate(duplicate_percentage_data):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.ylim(0, 100)
plt.tight_layout()
plt.show()


# ============================================================
# 6. DESCRIPTIVE STATISTICS AND POTENTIAL OUTLIERS
# ============================================================

# Select numerical variables relevant to the analysis
outlier_cols = [
    "lead_time",
    "adults",
    "babies",
    "days_in_waiting_list",
    "adr"
]

# Calculate minimum, maximum, and median values
outliers_df = pd.DataFrame({
    "Minimum": df[outlier_cols].min(),
    "Maximum": df[outlier_cols].max(),
    "Median": df[outlier_cols].median()
})

outliers_df.index.name = "Variable"

print("\nMinimum, Maximum and Median Values:")
print(outliers_df)


# Create boxplots to visually inspect potential outliers
selected = [
    "lead_time",
    "adr",
    "days_in_waiting_list",
    "total_of_special_requests"
]

plt.figure(figsize=(9, 5))

plt.boxplot(
    [df[col].dropna() for col in selected],
    tick_labels=selected
)

plt.ylabel("Value")
plt.title("Boxplots of Selected Numerical Variables")

plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# ============================================================
# 7. SKEWNESS ANALYSIS
# ============================================================

# Select all numerical columns
numeric_cols = df.select_dtypes(include="number").columns

# Calculate skewness for numerical variables
skewness = df[numeric_cols].skew()

# Select the six variables with the highest positive skewness
top_skewed = (
    skewness
    .sort_values(ascending=False)
    .head(6)
    .round(2)
)

print(f"\n{'Variable':<32} {'Skewness':>8}")
print("-" * 41)

for var, value in top_skewed.items():
    print(f"{var:<32} {value:>8.2f}")


# ============================================================
# 8. DISTRIBUTION ANALYSIS
# ============================================================

# Plot the distribution of booking lead time
plt.figure(figsize=(7, 5))

plt.hist(
    df["lead_time"],
    bins=40
)

plt.xlabel("Lead Time (Days)")
plt.ylabel("Frequency")
plt.title("Distribution of Booking Lead Time")

plt.tight_layout()
plt.show()


# Plot the distribution of Average Daily Rate (ADR)
plt.figure(figsize=(7, 5))

plt.hist(
    df["adr"],
    bins=50
)

plt.xlabel("Average Daily Rate")
plt.ylabel("Frequency")
plt.title("Distribution of Average Daily Rate")

plt.tight_layout()
plt.show()


# ============================================================
# 9. CORRELATION ANALYSIS
# ============================================================

# Select numerical variables for correlation analysis
numeric_df = df.select_dtypes(include="number")

# Calculate the Pearson correlation matrix
corr = numeric_df.corr()

# Display the correlation matrix as a heatmap
plt.figure(figsize=(12, 9))

plt.imshow(
    corr,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# 10. INCONSISTENT / INVALID RECORD CHECK
# ============================================================

# Identify bookings with zero adults but at least one child or baby
condition = (
    (df["adults"] == 0) &
    (
        (df["children"].fillna(0) > 0) |
        (df["babies"] > 0)
    )
)

# Count potentially inconsistent records
print(
    "\nPotentially inconsistent records:",
    condition.sum()
)