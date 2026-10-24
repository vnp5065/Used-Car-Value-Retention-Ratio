
import os
import pandas as pd


# ==========================================
# STEP 1: LOAD AND COMBINE DATASETS
# ==========================================

print("\n===== ORIGINAL DATASETS =====")

everylot = pd.read_csv("everylot_detailed.csv")
seeallcars = pd.read_csv("seeallcars_detailed.csv")

print("Everylot vehicles:", len(everylot))
print("SeeAllCars vehicles:", len(seeallcars))

combined = pd.concat(
    [everylot, seeallcars],
    ignore_index=True
)

print("\n===== COMBINED DATASET =====")
print("Total vehicles:", len(combined))
print("Total columns:", len(combined.columns))

print("\n===== COLUMN NAMES =====")
print(combined.columns.tolist())


# ==========================================
# STEP 2: INSPECT MISSING AND DUPLICATE DATA
# ==========================================

print("\n===== MISSING VALUES =====")
print(combined.isnull().sum())

combined["vin"] = (
    combined["vin"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

valid_vins = (
    combined["vin"].ne("")
    & combined["vin"].ne("UNKNOWN")
    & combined["vin"].ne("NAN")
)

duplicate_vins = (
    combined.loc[valid_vins, "vin"]
    .duplicated()
    .sum()
)

print("\n===== DUPLICATE CHECK =====")
print("Duplicate VINs:", duplicate_vins)
print("Missing VINs:", int((~valid_vins).sum()))


# ==========================================
# STEP 3: CHECK NUMERIC DATA
# ==========================================

print("\n===== NUMERIC DATA CHECK =====")

for column in ["year", "price", "mileage"]:

    combined[column] = pd.to_numeric(
        combined[column],
        errors="coerce"
    )

    print("\n", column)
    print("Missing:", combined[column].isnull().sum())
    print("Minimum:", combined[column].min())
    print("Maximum:", combined[column].max())


# ==========================================
# STEP 4: INSPECT VEHICLE DISTRIBUTION
# ==========================================

print("\n===== VEHICLES BY MAKE =====")
print(combined["make"].value_counts())

print("\n===== VEHICLES BY MODEL =====")
print(
    combined.groupby(["make", "model"]).size()
)

combined.to_csv(
    "combined_cars.csv",
    index=False
)

print("\nSaved to combined_cars.csv")


# ==========================================
# STEP 5: INSPECT SUSPICIOUS RECORDS
# ==========================================

print("\n===== VEHICLES WITH PRICES BELOW $1,000 =====")

low_price = combined[
    combined["price"] < 1000
]

print(
    low_price[
        [
            "year",
            "make",
            "model",
            "price",
            "mileage",
            "source",
            "listing_url"
        ]
    ].to_string(index=False)
)

print("Total:", len(low_price))


print("\n===== VEHICLES WITH MODEL YEAR 2027 =====")

future_year = combined[
    combined["year"] > 2026
]

print(
    future_year[
        [
            "year",
            "make",
            "model",
            "price",
            "mileage",
            "condition",
            "source"
        ]
    ].to_string(index=False)
)

print("Total:", len(future_year))


# ==========================================
# STEP 6: CLEAN DATASET
# ==========================================

print("\n===== DATA CLEANING =====")

cleaned = combined.copy()

# Remove duplicate valid VINs.
# Keep vehicles with missing VINs.

valid_clean_vins = (
    cleaned["vin"].ne("")
    & cleaned["vin"].ne("UNKNOWN")
    & cleaned["vin"].ne("NAN")
)

duplicate_valid_vins = (
    valid_clean_vins
    & cleaned["vin"].duplicated(keep="first")
)

cleaned = cleaned[
    ~duplicate_valid_vins
].copy()

# Remove duplicate listing URLs.
# Only deduplicate URLs that are present.

if "listing_url" in cleaned.columns:

    cleaned["listing_url"] = (
        cleaned["listing_url"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    valid_urls = cleaned["listing_url"].ne("")

    duplicate_urls = (
        valid_urls
        & cleaned["listing_url"].duplicated(
            keep="first"
        )
    )

    cleaned = cleaned[
        ~duplicate_urls
    ].copy()

# Remove invalid prices.
cleaned = cleaned[
    cleaned["price"] >= 1000
].copy()

# Standardize categorical columns.
categorical_columns = [
    "trim",
    "body",
    "engine",
    "transmission",
    "drivetrain",
    "fuel",
    "exterior",
    "interior",
    "location"
]

for column in categorical_columns:

    cleaned[column] = (
        cleaned[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

cleaned["year_needs_review"] = (
    cleaned["year"] > 2026
)

cleaned.to_csv(
    "cleaned_cars.csv",
    index=False
)

print("Original vehicles:", len(combined))
print("Cleaned vehicles:", len(cleaned))
print(
    "Removed vehicles:",
    len(combined) - len(cleaned)
)

print("\nVehicles by source:")
print(cleaned["source"].value_counts())

print(
    "\nVehicles requiring year review:",
    int(cleaned["year_needs_review"].sum())
)

print("\nSaved to cleaned_cars.csv")


# ==========================================
# STEP 7: MSRP MATCHING PREPARATION
# ==========================================

print("\n===== MSRP MATCHING PREPARATION =====")

cars = cleaned.copy()

cars["trim"] = cars["trim"].fillna("Unknown")

unique_configurations = cars[
    ["year", "make", "model", "trim"]
].drop_duplicates()

print("Total cleaned vehicles:", len(cars))
print(
    "Unique vehicle configurations:",
    len(unique_configurations)
)

missing_trim = cars[
    cars["trim"].str.upper() == "UNKNOWN"
]

print(
    "Vehicles with unknown trim:",
    len(missing_trim)
)

print("\n===== UNIQUE CONFIGURATIONS BY MAKE =====")
print(
    unique_configurations.groupby("make").size()
)

print("\n===== VEHICLES BY MODEL YEAR =====")
print(
    cars["year"].value_counts().sort_index()
)

print("\n===== MOST COMMON TRIMS =====")
print(
    cars["trim"].value_counts().head(20)
)

print("\n===== ORIGINAL MODEL NAMES =====")
print(
    cars.groupby(["make", "model"])["original_model"]
    .unique()
    .to_string()
)

print("\n===== ORIGINAL MODEL BREAKDOWN =====")

model_counts = (
    cars.groupby(
        ["make", "model", "original_model"]
    )
    .size()
    .reset_index(name="count")
)

print(model_counts.to_string(index=False))

print("\n===== VEHICLES WITH UNKNOWN TRIM BY MODEL =====")

unknown_trim = cars[
    cars["trim"].str.upper() == "UNKNOWN"
]

print(
    unknown_trim.groupby(
        ["make", "model"]
    ).size().to_string()
)


# ==========================================
# STEP 8: CORRECT MODEL GROUPING
# ==========================================

print("\n===== MODEL CORRECTION =====")

excluded_models = [
    "Corolla Cross",
    "Corolla Cross Hybrid",
    "Camry Solara",
    "Accord Crosstour",
    "F-150 Lightning"
]

corrected = cars[
    ~cars["original_model"].isin(excluded_models)
].copy()

print("Before correction:", len(cars))
print("After correction:", len(corrected))
print("Removed:", len(cars) - len(corrected))

print("\nVehicles by source:")
print(corrected["source"].value_counts())

print("\nVehicles by make and model:")
print(
    corrected.groupby(["make", "model"]).size()
)

corrected.to_csv(
    "corrected_cars.csv",
    index=False
)

print("\nSaved to corrected_cars.csv")


# ==========================================
# STEP 9: PREPARE MSRP LOOKUP
# ==========================================

print("\n===== MSRP LOOKUP PREPARATION =====")

cars = corrected.copy()

msrp_lookup = cars[
    [
        "year",
        "make",
        "model",
        "original_model",
        "trim",
        "drivetrain"
    ]
].drop_duplicates().copy()

msrp_lookup["original_msrp"] = pd.NA
msrp_lookup["msrp_source_url"] = ""
msrp_lookup["msrp_notes"] = ""

msrp_lookup = msrp_lookup.sort_values(
    ["make", "model", "year", "trim"]
)

msrp_lookup.to_csv(
    "msrp_lookup.csv",
    index=False
)

print("Total vehicles:", len(cars))
print(
    "Unique MSRP configurations:",
    len(msrp_lookup)
)

print("\nConfigurations by make:")
print(msrp_lookup["make"].value_counts())

print("\nSaved to msrp_lookup.csv")


# ==========================================
# STEP 10: STANDARDIZE MSRP CONFIGURATIONS
# ==========================================

print("\n===== MSRP STANDARDIZATION =====")

standardized = corrected.copy()

for column in [
    "make",
    "original_model",
    "trim",
    "drivetrain"
]:

    standardized[column] = (
        standardized[column]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

drivetrain_mapping = {
    "FRONT WHEEL DRIVE": "FWD",
    "FRONT-WHEEL DRIVE": "FWD",
    "ALL WHEEL DRIVE": "AWD",
    "ALL-WHEEL DRIVE": "AWD",
    "FOUR WHEEL DRIVE": "4WD",
    "FOUR-WHEEL DRIVE": "4WD",
    "4X4": "4WD",
    "REAR WHEEL DRIVE": "RWD",
    "REAR-WHEEL DRIVE": "RWD"
}

standardized["drivetrain"] = (
    standardized["drivetrain"]
    .replace(drivetrain_mapping)
)

simplified_msrp = standardized[
    [
        "year",
        "make",
        "model",
        "original_model",
        "trim",
        "drivetrain"
    ]
].drop_duplicates().copy()

simplified_msrp = simplified_msrp.sort_values(
    ["make", "model", "year", "trim"]
)

simplified_msrp["original_msrp"] = pd.NA
simplified_msrp["msrp_source_url"] = ""
simplified_msrp["msrp_notes"] = ""

simplified_msrp.to_csv(
    "msrp_lookup_standardized.csv",
    index=False
)

print(
    "Standardized configurations:",
    len(simplified_msrp)
)

print("\nSaved to msrp_lookup_standardized.csv")


# ==========================================
# STEP 11: CHECK MSRP MATCHING READINESS
# ==========================================

print("\n===== MSRP MATCHING READINESS =====")

review = corrected.copy()

for column in [
    "trim",
    "drivetrain",
    "original_model"
]:

    review[column] = (
        review[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

def check_msrp_status(row):

    if row["trim"].upper() in [
        "UNKNOWN", "NAN", ""
    ]:
        return "Missing trim"

    if (
        str(row["make"]).upper() == "FORD"
        and str(row["model"]).upper() == "F-150"
    ):
        return "Needs truck configuration review"

    if row["year"] >= 2027:
        return "Review model year"

    return "Ready for MSRP research"


review["msrp_status"] = review.apply(
    check_msrp_status,
    axis=1
)

review.to_csv(
    "msrp_match_review.csv",
    index=False
)

print("Total vehicles:", len(review))

print("\nMSRP matching status:")
print(review["msrp_status"].value_counts())

print("\nStatus by source:")
print(
    pd.crosstab(
        review["source"],
        review["msrp_status"]
    )
)

print("\nSaved to msrp_match_review.csv")


# ==========================================
# STEP 12: PRIORITIZE MSRP RESEARCH
# ==========================================

print("\n===== MSRP RESEARCH PRIORITIES =====")

ready = review[
    review["msrp_status"]
    == "Ready for MSRP research"
].copy()

ready["trim"] = (
    ready["trim"]
    .str.strip()
    .str.upper()
)

priorities = (
    ready.groupby(
        [
            "year",
            "make",
            "original_model",
            "trim"
        ],
        dropna=False
    )
    .size()
    .reset_index(name="vehicle_count")
    .sort_values(
        "vehicle_count",
        ascending=False
    )
)

priorities["original_msrp"] = pd.NA
priorities["msrp_source_url"] = ""
priorities["msrp_notes"] = ""

priorities.to_csv(
    "msrp_research_priorities.csv",
    index=False
)

print(
    "Vehicles ready for research:",
    len(ready)
)

print(
    "Unique research groups:",
    len(priorities)
)

print("\nTop 20 MSRP research priorities:")
print(
    priorities.head(20).to_string(index=False)
)

print("\nSaved to msrp_research_priorities.csv")


# ==========================================
# STEP 13: PRESERVE VERIFIED MSRP DATABASE
# ==========================================

print("\n===== VERIFIED MSRP DATABASE =====")

file_name = "msrp_verified.csv"

columns = [
    "year",
    "make",
    "original_model",
    "trim",
    "original_msrp",
    "msrp_source_url",
    "drivetrain"
]

if os.path.exists(file_name):

    verified = pd.read_csv(file_name)

    print("Existing MSRP database found.")

    missing_columns = [
        column for column in columns
        if column not in verified.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing MSRP columns: "
            + str(missing_columns)
        )

    for column in [
        "make",
        "original_model",
        "trim",
        "drivetrain"
    ]:

        verified[column] = (
            verified[column]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    verified["year"] = pd.to_numeric(
        verified["year"],
        errors="coerce"
    )

    verified["original_msrp"] = pd.to_numeric(
        verified["original_msrp"],
        errors="coerce"
    )

    keys = [
        "year",
        "make",
        "original_model",
        "trim",
        "drivetrain"
    ]

    conflicts = (
        verified.groupby(keys)["original_msrp"]
        .nunique()
    )

    if (conflicts > 1).any():

        print(
            conflicts[conflicts > 1]
        )

        raise ValueError(
            "Conflicting MSRP prices found. "
            "Review them before continuing."
        )

    verified = verified.drop_duplicates(
        subset=keys
    )

    print(
        "Existing MSRP configurations:",
        len(verified)
    )

else:

    verified = pd.DataFrame(columns=columns)

    print("No existing MSRP database found.")

verified.to_csv(
    file_name,
    index=False
)

print("\nMSRP database preserved successfully.")
print(
    "Total MSRP configurations:",
    len(verified)
)

print("\nSaved to msrp_verified.csv")


# ==========================================
# STEP 14: EXPAND VERIFIED MSRP DATABASE
# ==========================================

print("\n===== EXPANDING VERIFIED MSRP DATABASE =====")

msrp_file = "msrp_verified.csv"

msrp = pd.read_csv(msrp_file)

new_records = [
    {
        "year": 2024,
        "make": "HONDA",
        "original_model": "CIVIC",
        "trim": "SPORT",
        "original_msrp": 25550,
        "msrp_source_url": (
            "https://hondanews.com/en-US/releases/"
            "release-5003aaa39c009393f5d06d620f08fa05-"
            "2024-honda-civic-sedan-pricing-and-epa-ratings"
            "?downloadUrl=%2Fen-US%2Freleases%2F"
            "release-5003aaa39c009393f5d06d620f08fa05"
            "%2Fdownload&la=1"
        ),
        "drivetrain": "FWD"
    },
    {
        "year": 2023,
        "make": "HONDA",
        "original_model": "CR-V",
        "trim": "EX-L",
        "original_msrp": 35260,
        "msrp_source_url": (
            "https://hondanews.com/en-US/"
            "honda-automobiles/releases/"
            "release-1503019bd8a757ea08267d794438d875-"
            "2023-honda-cr-v-hybrid-pricing-and-epa-ratings"
        ),
        "drivetrain": "AWD"
    },
    {
        "year": 2023,
        "make": "HONDA",
        "original_model": "CR-V",
        "trim": "EX",
        "original_msrp": 32610,
        "msrp_source_url": (
            "https://hondanews.com/en-US/"
            "honda-automobiles/releases/"
            "release-1503019bd8a757ea08267d794438d875-"
            "2023-honda-cr-v-hybrid-pricing-and-epa-ratings"
        ),
        "drivetrain": "AWD"
    },
    {
        "year": 2023,
        "make": "HONDA",
        "original_model": "CR-V",
        "trim": "SPORT TOURING",
        "original_msrp": 38600,
        "msrp_source_url": (
            "https://hondanews.com/en-US/"
            "honda-automobiles/releases/"
            "release-1503019bd8a757ea08267d794438d875-"
            "2023-honda-cr-v-hybrid-pricing-and-epa-ratings"
        ),
        "drivetrain": "AWD"
    }
]

new_df = pd.DataFrame(new_records)

keys = [
    "year",
    "make",
    "original_model",
    "trim",
    "drivetrain"
]

for col in keys[1:]:
    msrp[col] = msrp[col].astype(str).str.strip().str.upper()
    new_df[col] = new_df[col].astype(str).str.strip().str.upper()

msrp["year"] = pd.to_numeric(msrp["year"], errors="raise")
new_df["year"] = pd.to_numeric(new_df["year"], errors="raise")

# Do not overwrite any existing MSRP configurations.
existing_keys = pd.MultiIndex.from_frame(msrp[keys])
new_keys = pd.MultiIndex.from_frame(new_df[keys])

records_to_add = new_df[
    ~new_keys.isin(existing_keys)
].copy()

msrp = pd.concat(
    [msrp, records_to_add],
    ignore_index=True
)

msrp.to_csv(msrp_file, index=False)

print("New configurations added:", len(records_to_add))
print("Total MSRP configurations:", len(msrp))
print("\nSaved to msrp_verified.csv")
