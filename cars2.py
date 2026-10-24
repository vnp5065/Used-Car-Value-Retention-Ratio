
import requests
import pandas as pd
import time

BASE_URL = "https://seeallcars.com/api/search"

CSV_FILE = "seeallcars_detailed.csv"
EVERYLOT_FILE = "everylot_detailed.csv"

TARGET_VEHICLES = 550
MAX_PAGES = 100

headers = {
    "User-Agent": "AcademicVehicleResearch/1.0"
}

selected_models = {
    "Toyota": ["Camry", "Corolla", "RAV4"],
    "Honda": ["Civic", "Accord", "CR-V"],
    "Ford": ["F-150", "Explorer", "Escape"]
}

# Distribute vehicles across the three makes
make_targets = {
    "Toyota": 185,
    "Honda": 185,
    "Ford": 180
}

columns = [
    "year", "make", "model", "original_model",
    "trim", "body", "condition", "mileage",
    "engine", "transmission", "drivetrain",
    "fuel", "exterior", "interior", "vin",
    "price", "location", "listing_url", "source"
]

cars = []
seen_vins = set()
seen_urls = set()
everylot_vins = set()


# NORMALIZE MODEL NAMES
def normalize_model(make, model):

    if make not in selected_models:
        return None

    model = str(model).strip().lower()

    for selected in selected_models[make]:

        selected_lower = selected.lower()

        if (
            model == selected_lower
            or model.startswith(selected_lower + " ")
        ):
            return selected

    return None


# LOAD EVERYLOT VINS
def load_everylot_vins():

    try:
        data = pd.read_csv(
            EVERYLOT_FILE,
            dtype=str
        ).fillna("")

        for vin in data["vin"]:

            vin = vin.strip().upper()

            if vin:
                everylot_vins.add(vin)

        print("Everylot VINs loaded:", len(everylot_vins))

    except (FileNotFoundError, KeyError):
        print("Warning: Could not load Everylot VINs.")
        print("Cross-source duplicate checking is unavailable.")


# LOAD PREVIOUS SEEALLCARS DATA
def load_existing_data():

    try:
        data = pd.read_csv(
            CSV_FILE,
            dtype=str
        ).fillna("")

        for _, row in data.iterrows():

            car = {
                column: row.get(column, "")
                for column in columns
            }

            vin = car["vin"].strip().upper()
            url = car["listing_url"].strip()

            if not vin or not url:
                continue

            if vin in everylot_vins:
                continue

            if vin in seen_vins or url in seen_urls:
                continue

            if car["make"] not in selected_models:
                continue

            if normalize_model(
                car["make"], car["model"]
            ) is None:
                continue

            cars.append(car)

            seen_vins.add(vin)
            seen_urls.add(url)

        print("Previously saved SeeAllCars vehicles:", len(cars))

    except FileNotFoundError:
        print("Starting new SeeAllCars dataset.")


# COUNT VEHICLES FOR A MAKE
def count_make(make):

    return sum(
        1 for car in cars
        if car["make"] == make
    )


# SAVE DATA
def save_data():

    data = pd.DataFrame(
        cars,
        columns=columns
    )

    data.to_csv(
        CSV_FILE,
        index=False
    )


# PROCESS VEHICLES FROM API
def process_results(results, selected_make):

    added = 0

    for vehicle in results:

        if len(cars) >= TARGET_VEHICLES:
            break

        if count_make(selected_make) >= make_targets[selected_make]:
            break

        make = str(vehicle.get("make") or "").strip()

        if make != selected_make:
            continue

        original_model = str(
            vehicle.get("model") or ""
        ).strip()

        model = normalize_model(
            make,
            original_model
        )

        if model is None:
            continue

        condition = str(
            vehicle.get("condition") or ""
        ).lower()

        if condition != "used":
            continue

        # Use actual listed prices, not estimates
        if vehicle.get("price_kind") != "listed":
            continue

        if vehicle.get("price_suspect"):
            continue

        price = vehicle.get("price")
        mileage = vehicle.get("mileage")
        year = vehicle.get("year")

        try:
            price = float(price)
            mileage = int(mileage)
            year = int(year)

        except (TypeError, ValueError):
            continue

        if price <= 0 or mileage < 0:
            continue

        if year < 1990 or year > 2026:
            continue

        vin = str(
            vehicle.get("vin") or ""
        ).strip().upper()

        listing_url = str(
            vehicle.get("url") or ""
        ).strip()

        # VIN required for reliable deduplication
        if len(vin) != 17:
            continue

        if not listing_url:
            continue

        if vin in seen_vins:
            continue

        if vin in everylot_vins:
            continue

        if listing_url in seen_urls:
            continue

        city = str(vehicle.get("city") or "").strip()
        state = str(vehicle.get("state") or "").strip()

        location = ", ".join(
            part for part in [city, state]
            if part
        )

        car = {
            "year": year,
            "make": make,
            "model": model,
            "original_model": original_model,
            "trim": vehicle.get("trim") or "",
            "body": vehicle.get("body_style") or "",
            "condition": "Used",
            "mileage": mileage,
            "engine": "",
            "transmission": vehicle.get("transmission") or "",
            "drivetrain": vehicle.get("drivetrain") or "",
            "fuel": vehicle.get("fuel") or "",
            "exterior": vehicle.get("ext_color") or "",
            "interior": "",
            "vin": vin,
            "price": price,
            "location": location,
            "listing_url": listing_url,
            "source": "SeeAllCars"
        }

        cars.append(car)

        seen_vins.add(vin)
        seen_urls.add(listing_url)

        added += 1

    return added


# MAIN PROGRAM
load_everylot_vins()
load_existing_data()

session = requests.Session()
session.headers.update(headers)

for make in selected_models:

    if len(cars) >= TARGET_VEHICLES:
        break

    if count_make(make) >= make_targets[make]:
        continue

    print("\n============================")
    print("SCRAPING:", make)
    print("============================")

    previous_page_vins = None

    for page in range(1, MAX_PAGES + 1):

        if len(cars) >= TARGET_VEHICLES:
            break

        if count_make(make) >= make_targets[make]:
            break

        params = {
            "make": make,
            "condition": "used",
            "page": page,
            "sort": "newest"
        }

        try:
            response = session.get(
                BASE_URL,
                params=params,
                timeout=30
            )

            if response.status_code != 200:

                print(
                    "Page:", page,
                    "| Status:", response.status_code
                )

                if response.status_code in [403, 429]:
                    print("Access restricted. Stopping.")
                    break

                break

            data = response.json()

            results = data.get("results", [])

            if not results:
                print("No more listings for", make)
                break

            page_vins = {
                str(car.get("vin"))
                for car in results
                if car.get("vin")
            }

            if page_vins == previous_page_vins:
                print("Repeated page. Stopping", make)
                break

            previous_page_vins = page_vins

            added = process_results(
                results,
                make
            )

            save_data()

            print(
                "Make:", make,
                "| Page:", page,
                "| New:", added,
                "| Make total:", count_make(make),
                "| Progress:",
                str(len(cars)) + "/" + str(TARGET_VEHICLES)
            )

            time.sleep(2)

        except requests.RequestException as e:
            print("Request error:", e)
            break

        except ValueError:
            print("Invalid JSON response.")
            break


# FINAL RESULTS
save_data()

data = pd.DataFrame(
    cars,
    columns=columns
)

print("\n============================")
print("FINAL RESULTS")
print("============================")

print("Total vehicles:", len(data))
print("Total columns:", len(data.columns))

print("\nVehicles by make:")
print(data["make"].value_counts())

print("\nVehicles by make and model:")
print(
    data.groupby(
        ["make", "model"]
    ).size()
)

print("\nDuplicate VINs:")
print(data["vin"].duplicated().sum())

print("\nSaved to:", CSV_FILE)

if len(data) >= TARGET_VEHICLES:
    print("\nTarget of 550 vehicles reached!")
else:
    print(
        "\nCollected",
        len(data),
        "out of",
        TARGET_VEHICLES
    )

print("\nProcess completed.")
