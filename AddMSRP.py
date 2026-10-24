
import pandas as pd

filename = "msrp_verified.csv"

msrp = pd.read_csv(filename)

print("===== BEFORE UPDATE =====")
print("Existing MSRP configurations:", len(msrp))

new_prices = [
    [
        2026, "Toyota", "Corolla", "LE", 22725,
        "https://pressroom.toyota.com/bold-and-compact-the-2026-toyota-corolla-offers-impressive-features-and-technology/",
        "FWD"
    ],
    [
        2024, "Honda", "CR-V", "EX", 32010,
        "https://automobiles.honda.com/2024/cr-v/",
        "FWD"
    ],
    [
        2024, "Honda", "CR-V", "EX-L", 34660,
        "https://automobiles.honda.com/2024/cr-v/",
        "FWD"
    ],
    [
        2025, "Honda", "CR-V", "EX", 32350,
        "https://automobiles.honda.com/tools/build-and-price-trimwalk?modelseries=cr-v",
        "FWD"
    ],
    [
        2025, "Honda", "CR-V", "EX-L", 35000,
        "https://automobiles.honda.com/tools/build-and-price-trimwalk?modelseries=cr-v",
        "FWD"
    ]
]

columns = [
    "year",
    "make",
    "original_model",
    "trim",
    "original_msrp",
    "msrp_source_url",
    "drivetrain"
]

new_df = pd.DataFrame(new_prices, columns=columns)

keys = [
    "year",
    "make",
    "original_model",
    "trim",
    "drivetrain"
]

combined = pd.concat([msrp, new_df], ignore_index=True)

for col in ["make", "original_model", "trim", "drivetrain"]:
    combined[col] = (
        combined[col].astype("string").str.strip().str.upper()
    )

conflicts = combined.groupby(keys, dropna=False)[
    "original_msrp"
].nunique()

if (conflicts > 1).any():
    raise ValueError("Conflicting MSRP values. Review before saving.")

combined = combined.drop_duplicates(subset=keys, keep="first")

combined.to_csv(filename, index=False)

print("\n===== AFTER UPDATE =====")
print("Total MSRP configurations:", len(combined))
print("New configurations added:", len(combined) - len(msrp))
print("\nSaved successfully to", filename)
