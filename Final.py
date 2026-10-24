
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# STEP 1: LOAD DATA
# ==========================================

print("\n===== LOADING DATA =====")

cars = pd.read_csv("corrected_cars.csv")
msrp = pd.read_csv("msrp_verified.csv")

new_msrp = pd.DataFrame([
    {
        "year": 2020,
        "make": "Toyota",
        "original_model": "Corolla",
        "trim": "LE",
        "original_msrp": 20050,
        "msrp_source_url": "https://www.cars.com/research/toyota-corolla-2020/trims/",
        "drivetrain": "FWD"
    },
    {
        "year": 2021,
        "make": "Toyota",
        "original_model": "Corolla",
        "trim": "LE",
        "original_msrp": 20475,
        "msrp_source_url": "https://www.cars.com/research/toyota-corolla-2021/trims/",
        "drivetrain": "FWD"
    },
    {
        "year": 2024,
        "make": "Honda",
        "original_model": "CR-V",
        "trim": "EX-L",
        "original_msrp": 36160,
        "msrp_source_url": (
            "https://hondanews.com/en-US/releases/"
            "release-a463299e9046a088b84018a7580622d8-"
            "2024-honda-cr-v-pricing-and-epa-ratings-2"
        ),
        "drivetrain": "AWD"
    }
])

msrp = pd.concat([msrp, new_msrp], ignore_index=True)

print("Vehicle listings:", len(cars))
print("MSRP configurations loaded:", len(msrp))


# ==========================================
# STEP 2: MERGE DEALERSHIP DESCRIPTIONS
# ==========================================

print("\n===== MERGING DEALERSHIP DESCRIPTIONS =====")

desc = pd.read_csv(
    "seeallcars_detailed_with_descriptions.csv"
)

required_columns = [
    "listing_url",
    "description",
    "well_maintained",
    "backup_camera"
]

for col in required_columns:
    if col not in desc.columns:
        raise ValueError(
            f"Missing column in description file: {col}"
        )

if "listing_url" not in cars.columns:
    raise ValueError(
        "corrected_cars.csv must contain listing_url."
    )


def clean_url(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.rstrip("/")
        .str.lower()
    )


cars["url_key"] = clean_url(cars["listing_url"])
desc["url_key"] = clean_url(desc["listing_url"])

desc = desc[
    desc["url_key"] != ""
].drop_duplicates(subset="url_key")

desc = desc[
    [
        "url_key",
        "description",
        "well_maintained",
        "backup_camera"
    ]
]

# Avoid duplicate column names if the cleaned file
# already contains description-related columns.
cars = cars.drop(
    columns=[
        "description",
        "well_maintained",
        "backup_camera",
        "description_available"
    ],
    errors="ignore"
)

original_count = len(cars)

cars = cars.merge(
    desc,
    on="url_key",
    how="left",
    validate="many_to_one"
)

if len(cars) != original_count:
    raise ValueError(
        "Description merge changed the vehicle count."
    )

cars.drop(columns=["url_key"], inplace=True)

cars["description"] = (
    cars["description"]
    .fillna("")
    .astype(str)
)

cars["description_available"] = (
    cars["description"].str.strip().ne("")
).astype(int)

for col in ["well_maintained", "backup_camera"]:
    cars[col] = (
        pd.to_numeric(cars[col], errors="coerce")
        .fillna(0)
        .astype(int)
    )

print("Total vehicles:", len(cars))
print(
    "Vehicles with descriptions:",
    int(cars["description_available"].sum())
)
print(
    "Well-maintained mentions:",
    int(cars["well_maintained"].sum())
)
print(
    "Backup-camera mentions:",
    int(cars["backup_camera"].sum())
)


# ==========================================
# STEP 3: STANDARDIZE DATA
# ==========================================

print("\n===== DATA STANDARDIZATION =====")

keys = [
    "year",
    "make",
    "original_model",
    "trim",
    "drivetrain"
]


def standardize_drivetrain(series):
    values = (
        series.fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return values.replace({
        "FRONT WHEEL DRIVE": "FWD",
        "FRONT-WHEEL DRIVE": "FWD",
        "FRONT-WHEEL DRIVE (FWD)": "FWD",
        "ALL WHEEL DRIVE": "AWD",
        "ALL-WHEEL DRIVE": "AWD",
        "FOUR WHEEL DRIVE": "4WD",
        "FOUR-WHEEL DRIVE": "4WD",
        "4X4": "4WD",
        "4WD/4-WHEEL DRIVE/4X4": "4WD",
        "REAR WHEEL DRIVE": "RWD",
        "REAR-WHEEL DRIVE": "RWD"
    })


for df in [cars, msrp]:

    for col in ["make", "original_model", "trim"]:
        df[col] = (
            df[col]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    df["drivetrain"] = standardize_drivetrain(
        df["drivetrain"]
    )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

cars["price"] = pd.to_numeric(
    cars["price"],
    errors="coerce"
)

cars["mileage"] = pd.to_numeric(
    cars["mileage"],
    errors="coerce"
)

msrp["original_msrp"] = pd.to_numeric(
    msrp["original_msrp"],
    errors="coerce"
)

msrp = msrp[
    msrp["original_msrp"] > 0
].copy()

conflicts = (
    msrp.groupby(keys)["original_msrp"]
    .nunique()
)

if (conflicts > 1).any():
    print(conflicts[conflicts > 1])
    raise ValueError(
        "Conflicting MSRP values for the same configuration."
    )

msrp = msrp.drop_duplicates(subset=keys)

print("Unique MSRP configurations:", len(msrp))


# ==========================================
# STEP 4: CORRECT DRIVETRAIN ERRORS
# ==========================================

print("\n===== CORRECTING DRIVETRAIN ERRORS =====")

cars["original_drivetrain"] = cars["drivetrain"]

# Model-based assumptions; not VIN-verified.

corolla_mask = (
    (cars["make"] == "TOYOTA")
    & (cars["original_model"] == "COROLLA")
    & (cars["year"].between(2014, 2026))
    & cars["trim"].isin([
        "L", "LE", "SE", "XLE", "XSE", "FX"
    ])
    & cars["drivetrain"].isin([
        "4WD", "AWD", "NOT LISTED"
    ])
)

civic_mask = (
    (cars["make"] == "HONDA")
    & (cars["original_model"] == "CIVIC")
    & (cars["year"].between(2016, 2026))
    & cars["trim"].isin([
        "LX", "EX", "EX-L", "SPORT", "TOURING"
    ])
    & cars["drivetrain"].isin([
        "4WD", "AWD", "NOT LISTED"
    ])
)

print(
    "Corolla records corrected:",
    int(corolla_mask.sum())
)

print(
    "Civic records corrected:",
    int(civic_mask.sum())
)

cars.loc[corolla_mask, "drivetrain"] = "FWD"
cars.loc[civic_mask, "drivetrain"] = "FWD"

cars["drivetrain_corrected"] = (
    cars["original_drivetrain"] != cars["drivetrain"]
)

print(
    "Total drivetrain corrections:",
    int(cars["drivetrain_corrected"].sum())
)


# ==========================================
# STEP 5: MSRP MATCHING
# ==========================================

print("\n===== MSRP MATCHING =====")

cars = cars.merge(
    msrp[
        keys + [
            "original_msrp",
            "msrp_source_url"
        ]
    ],
    on=keys,
    how="left",
    validate="many_to_one"
)

cars["msrp_match_eligible"] = (
    cars["trim"].ne("UNKNOWN")
    & cars["drivetrain"].ne("UNKNOWN")
    & cars["drivetrain"].ne("NOT LISTED")
    & cars["original_msrp"].notna()
    & cars["price"].gt(0)
    & cars["year"].lt(2027)
    & ~(
        (cars["make"] == "FORD")
        & (
            cars["model"]
            .fillna("")
            .astype(str)
            .str.upper() == "F-150"
        )
    )
)

cars["value_retention_ratio"] = np.nan

mask = cars["msrp_match_eligible"]

cars.loc[mask, "value_retention_ratio"] = (
    cars.loc[mask, "price"]
    / cars.loc[mask, "original_msrp"]
)

print("Total vehicles:", len(cars))
print(
    "Vehicles with matched MSRP:",
    int(mask.sum())
)
print(
    "Vehicles without eligible MSRP:",
    int((~mask).sum())
)


# ==========================================
# STEP 6: MSRP RESEARCH PRIORITIES
# ==========================================

print("\n===== MSRP RESEARCH PRIORITIES =====")

eligible = cars[
    (cars["trim"] != "UNKNOWN")
    & (cars["drivetrain"] != "UNKNOWN")
    & (cars["drivetrain"] != "NOT LISTED")
    & (cars["year"] < 2027)
    & (cars["price"] > 0)
    & ~(
        (cars["make"] == "FORD")
        & (
            cars["model"]
            .fillna("")
            .astype(str)
            .str.upper() == "F-150"
        )
    )
].copy()

missing = eligible[
    eligible["original_msrp"].isna()
].copy()

priority = (
    missing.groupby(keys)
    .size()
    .reset_index(name="vehicle_count")
    .sort_values(
        "vehicle_count",
        ascending=False
    )
    .reset_index(drop=True)
)

priority.insert(
    0,
    "priority_rank",
    range(1, len(priority) + 1)
)

print("\nTop 30 missing MSRP configurations:")
print(priority.head(30).to_string(index=False))

priority.to_csv(
    "msrp_priority_by_drivetrain.csv",
    index=False
)

print("\nPotential coverage:")

for count in [5, 10, 20, 30, 50]:
    potential = (
        priority.head(count)["vehicle_count"].sum()
    )

    print(
        f"Top {count} configurations: "
        f"up to {potential} additional vehicles"
    )


# ==========================================
# STEP 7: FEATURE ENGINEERING
# ==========================================

print("\n===== FEATURE ENGINEERING =====")

cars["vehicle_age"] = (
    2026 - cars["year"]
).clip(lower=0)

cars["mileage_per_year"] = (
    cars["mileage"]
    / cars["vehicle_age"].clip(lower=1)
)

print(
    "Vehicles with descriptions:",
    int(cars["description_available"].sum())
)

print(
    "Well-maintained mentions:",
    int(cars["well_maintained"].sum())
)

print(
    "Backup-camera mentions:",
    int(cars["backup_camera"].sum())
)


# ==========================================
# STEP 8: SAVE INTEGRATED DATA
# ==========================================

cars.to_csv(
    "final_integrated_cars.csv",
    index=False
)

print("\nSaved: final_integrated_cars.csv")


# ==========================================
# STEP 9: EXPLORATORY DATA ANALYSIS
# ==========================================

print("\n===== EXPLORATORY DATA ANALYSIS =====")

os.makedirs("graphs", exist_ok=True)

sns.set_theme(style="whitegrid")

# Graph 1: Price distribution

plt.figure(figsize=(9, 5))

sns.histplot(
    cars["price"].dropna(),
    bins=35
)

plt.title("Distribution of Used Car Listing Prices")
plt.xlabel("Listing Price ($)")
plt.ylabel("Number of Vehicles")
plt.tight_layout()

plt.savefig(
    "graphs/price_distribution.png",
    dpi=150
)
plt.close()


# Graph 2: Average price by make

plt.figure(figsize=(8, 5))

sns.barplot(
    data=cars,
    x="make",
    y="price",
    estimator=np.mean,
    errorbar=None
)

plt.title("Average Used Car Price by Make")
plt.xlabel("Make")
plt.ylabel("Average Price ($)")
plt.tight_layout()

plt.savefig(
    "graphs/average_price_by_make.png",
    dpi=150
)
plt.close()


# Graph 3: Mileage versus price

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=cars,
    x="mileage",
    y="price",
    hue="make",
    alpha=0.6
)

plt.title("Mileage vs Used Car Listing Price")
plt.xlabel("Mileage")
plt.ylabel("Listing Price ($)")
plt.tight_layout()

plt.savefig(
    "graphs/mileage_vs_price.png",
    dpi=150
)
plt.close()


# Graph 4: Value retention ratio

labeled = cars[
    cars["value_retention_ratio"].notna()
].copy()

if len(labeled) > 0:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        labeled["value_retention_ratio"],
        bins=20
    )

    plt.title("Distribution of Value Retention Ratios")
    plt.xlabel("Value Retention Ratio")
    plt.ylabel("Number of Vehicles")
    plt.tight_layout()

    plt.savefig(
        "graphs/value_retention.png",
        dpi=150
    )
    plt.close()

print("Graphs saved in the graphs folder.")


# ==========================================
# STEP 10: MACHINE LEARNING
# ==========================================

print("\n===== MACHINE LEARNING =====")

labeled = labeled[
    labeled["value_retention_ratio"] > 0
].copy()

print(
    "Labeled vehicles available:",
    len(labeled)
)

MIN_LABELED_SAMPLES = 100

if len(labeled) < MIN_LABELED_SAMPLES:

    print("\nNot enough labeled vehicles for training.")
    print(
        "Minimum set in this pipeline:",
        MIN_LABELED_SAMPLES
    )
    print(
        "Add more accurate MSRP configurations and rerun."
    )

else:

    numeric_features = [
        "year",
        "mileage",
        "vehicle_age",
        "mileage_per_year",
        "well_maintained",
        "backup_camera",
        "description_available"
    ]

    categorical_features = [
        "make",
        "model",
        "trim",
        "body",
        "transmission",
        "drivetrain",
        "fuel"
    ]

    features = (
        numeric_features + categorical_features
    )

    print("\n===== DESCRIPTION FEATURE COVERAGE =====")

    print(
        "Labeled vehicles with descriptions:",
        int(labeled["description_available"].sum())
    )

    print(
        "Labeled well-maintained mentions:",
        int(labeled["well_maintained"].sum())
    )

    print(
        "Labeled backup-camera mentions:",
        int(labeled["backup_camera"].sum())
    )

    X = labeled[features].copy()

    for col in categorical_features:
        X[col] = (
            X[col]
            .fillna("Unknown")
            .astype(str)
        )

    y = labeled[
        "value_retention_ratio"
    ].astype(float)

    numeric_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ])

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    results = []
    fitted_models = {}

    for name, model in models.items():

        pipeline = Pipeline([
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                model
            )
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        fitted_models[name] = pipeline

        print("\nModel:", name)
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2:", round(r2, 4))

    # ======================================
    # STEP 11: MODEL COMPARISON
    # ======================================

    print("\n===== MODEL COMPARISON =====")

    results_df = pd.DataFrame(results)

    print(results_df.to_string(index=False))

    results_df.to_csv(
        "model_comparison.csv",
        index=False
    )

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=results_df,
        x="Model",
        y="RMSE"
    )

    plt.title("Regression Model Comparison")
    plt.ylabel("RMSE (Lower Is Better)")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        "graphs/model_comparison.png",
        dpi=150
    )
    plt.close()


    # ======================================
    # STEP 12: FEATURE IMPORTANCE
    # ======================================

    print("\n===== FEATURE IMPORTANCE =====")

    best_model_name = (
        results_df.sort_values("RMSE")
        .iloc[0]["Model"]
    )

    best_pipeline = fitted_models[
        best_model_name
    ]

    print("Best model:", best_model_name)

    importance = permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        scoring="neg_mean_squared_error",
        n_repeats=5,
        random_state=42
    )

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance.importances_mean
    }).sort_values(
        "Importance",
        ascending=False
    )

    print(
        importance_df.to_string(index=False)
    )

    importance_df.to_csv(
        "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=importance_df.head(10),
        x="Importance",
        y="Feature"
    )

    plt.title(
        "Top 10 Features by Permutation Importance"
    )
    plt.tight_layout()

    plt.savefig(
        "graphs/feature_importance.png",
        dpi=150
    )
    plt.close()

    print("\nSaved: model_comparison.csv")
    print("Saved: feature_importance.csv")


print("\n===== PIPELINE COMPLETED =====")
