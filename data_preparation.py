import pandas as pd
import numpy as np
import os


# ==========================================
# 1. FIND CSV FILE
# ==========================================

project_folder = os.path.dirname(os.path.abspath(__file__))

csv_files = [
    file for file in os.listdir(project_folder)
    if file.lower().endswith(".csv")
]

print("======================================")
print("NASSAU CANDY PROJECT")
print("======================================")

print("\nCSV files found:")

for file in csv_files:
    print("-", file)


# ==========================================
# 2. CHECK CSV FILE
# ==========================================

if len(csv_files) == 0:
    print("\nERROR: No CSV file found!")
    print("\nPlease put your original Nassau Candy CSV file")
    print("inside this folder:")
    print(project_folder)

    input("\nPress Enter to close...")
    exit()


# ==========================================
# 3. LOAD CSV FILE
# ==========================================

csv_path = os.path.join(
    project_folder,
    csv_files[0]
)

print("\nLoading file:")
print(csv_files[0])

df = pd.read_csv(
    csv_path,
    encoding="latin1"
)

print("\nDataset loaded successfully!")


# ==========================================
# 4. BASIC INFORMATION
# ==========================================

print("\n======================================")
print("DATASET INFORMATION")
print("======================================")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 5. FIRST 5 ROWS
# ==========================================

print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
# 6. MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 7. DUPLICATE ROWS
# ==========================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ==========================================
# 8. CONVERT DATE COLUMNS
# ==========================================

if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        errors="coerce"
    )

if "Ship Date" in df.columns:
    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        errors="coerce"
    )


# ==========================================
# 9. CREATE SHIPPING DAYS
# ==========================================

if "Order Date" in df.columns and "Ship Date" in df.columns:

    df["Shipping Days"] = (
        df["Ship Date"] - df["Order Date"]
    ).dt.days

else:

    print("\nERROR: Order Date or Ship Date column not found.")
    input("\nPress Enter to close...")
    exit()


# ==========================================
# 10. CONVERT NUMERICAL COLUMNS
# ==========================================

numeric_columns = [
    "Sales",
    "Units",
    "Gross Profit",
    "Cost"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ==========================================
# 11. CREATE PROFIT MARGIN
# ==========================================

if "Sales" in df.columns and "Gross Profit" in df.columns:

    df["Profit Margin"] = np.where(
        df["Sales"] != 0,
        (df["Gross Profit"] / df["Sales"]) * 100,
        0
    )


# ==========================================
# 12. REMOVE INVALID SHIPPING DAYS
# ==========================================

df = df[
    df["Shipping Days"].notna()
]

df = df[
    df["Shipping Days"] >= 0
]


# ==========================================
# 13. REMOVE EXTREME OUTLIERS
# ==========================================

Q1 = df["Shipping Days"].quantile(0.25)

Q3 = df["Shipping Days"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - (1.5 * IQR)

upper_limit = Q3 + (1.5 * IQR)

df = df[
    (df["Shipping Days"] >= lower_limit)
    &
    (df["Shipping Days"] <= upper_limit)
]


# ==========================================
# 14. SHIPPING STATISTICS
# ==========================================

print("\n======================================")
print("SHIPPING DAYS STATISTICS")
print("======================================")

print(df["Shipping Days"].describe())


# ==========================================
# 15. FINAL DATASET INFORMATION
# ==========================================

print("\n======================================")
print("FINAL CLEANED DATASET")
print("======================================")

print("Rows:", len(df))

print("Columns:", len(df.columns))


# ==========================================
# 16. SAVE CLEANED DATASET
# ==========================================

output_path = os.path.join(
    project_folder,
    "cleaned_nassau_candy.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nCleaned dataset created successfully!")

print("File:")
print(output_path)


# ==========================================
# 17. COMPLETED
# ==========================================

print("\n======================================")
print("DATA PREPARATION COMPLETED!")
print("======================================")