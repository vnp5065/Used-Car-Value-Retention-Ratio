
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

BASE_URL = "https://everylot.com"
CSV_FILE = "everylot_detailed.csv"

TARGET_VEHICLES = 500
MAX_PAGES = 20

headers = {
    "User-Agent": "AcademicVehicleResearch/1.0"
}

selected_models = {
    "Toyota": ["Camry", "Corolla", "RAV4"],
    "Honda": ["Civic", "Accord", "CR-V"],
    "Ford": ["F-150", "Explorer", "Escape"]
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


# LOAD PREVIOUSLY COLLECTED DATA
def load_existing_data():

    global cars

    try:
        data = pd.read_csv(
            CSV_FILE,
            dtype=str
        ).fillna("")

        for _, row in data.iterrows():

            car = row.to_dict()

            vin = str(
                car.get("vin", "")
            ).strip().upper()

            url = str(
                car.get("listing_url", "")
            ).strip()

            if vin and vin in seen_vins:
                continue

            if url and url in seen_urls:
                continue

            if not url:
                continue

            cars.append(car)
            seen_urls.add(url)

            if vin:
                seen_vins.add(vin)

        print("Previously saved vehicles:", len(cars))

    except FileNotFoundError:
        print("No previous CSV found. Starting fresh.")


# NORMALIZE MODEL NAMES
def normalize_model(make, model):

    if make not in selected_models:
        return None

    model = model.strip().lower()

    for selected in selected_models[make]:

        selected_lower = selected.lower()

        if (
            model == selected_lower
            or model.startswith(selected_lower + " ")
        ):
            return selected

    return None


# SCRAPE INDIVIDUAL VEHICLE DETAILS
def scrape_vehicle_details(url):

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        details = {}

        for item in soup.find_all("dt"):

            value = item.find_next_sibling("dd")

            if value is not None:

                key = item.get_text(
                    " ", strip=True
                ).lower()

                details[key] = value.get_text(
                    " ", strip=True
                )

        return details

    except requests.RequestException as e:
        print("Detail request error:", e)
        return None


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

    print("Total saved vehicles:", len(data))


# SCRAPE ONE PAGE
def scrape_page(soup, selected_make):

    vehicle_elements = soup.find_all(
        "li",
        class_="result-tile"
    )

    added = 0

    for vehicle in vehicle_elements:

        if len(cars) >= TARGET_VEHICLES:
            break

        title_element = vehicle.find(
            "a",
            class_="row-title"
        )

        if title_element is None:
            continue

        title = title_element.get_text(
            " ", strip=True
        )

        title_match = re.match(
            r"^(19\d{2}|20\d{2})\s+"
            r"(Toyota|Honda|Ford)\s+(.+)",
            title,
            re.IGNORECASE
        )

        if title_match is None:
            continue

        make = title_match.group(2)
        title_model = title_match.group(3)

        if make.lower() != selected_make.lower():
            continue

        if normalize_model(make, title_model) is None:
            continue

        listing_text = vehicle.get_text(
            " ", strip=True
        )

        mileage_match = re.search(
            r"([\d,]+)\s*mi\b",
            listing_text,
            re.IGNORECASE
        )

        price_matches = re.findall(
            r"\$([\d,]+)",
            listing_text
        )

        if mileage_match is None or not price_matches:
            continue

        mileage = int(
            mileage_match.group(1).replace(",", "")
        )

        price = int(
            price_matches[-1].replace(",", "")
        )

        listing_url = title_element.get("href", "")

        if listing_url.startswith("/"):
            listing_url = BASE_URL + listing_url

        if not listing_url or listing_url in seen_urls:
            continue

        details = scrape_vehicle_details(
            listing_url
        )

        if details is None:
            continue

        condition = details.get(
            "condition", ""
        ).strip()

        if condition.lower() != "used":
            continue

        detail_make = details.get(
            "make", ""
        ).strip()

        original_model = details.get(
            "model", ""
        ).strip()

        model = normalize_model(
            detail_make,
            original_model
        )

        if model is None:
            continue

        vin = details.get(
            "vin", ""
        ).strip().upper()

        if vin and vin in seen_vins:
            continue

        car = {
            "year": details.get("year", ""),
            "make": detail_make,
            "model": model,
            "original_model": original_model,
            "trim": details.get("trim", ""),
            "body": details.get("body", ""),
            "condition": condition,
            "mileage": mileage,
            "engine": details.get("engine", ""),
            "transmission": details.get("transmission", ""),
            "drivetrain": details.get("drivetrain", ""),
            "fuel": details.get("fuel", ""),
            "exterior": details.get("exterior", ""),
            "interior": details.get("interior", ""),
            "vin": vin,
            "price": price,
            "location": "",
            "listing_url": listing_url,
            "source": "Everylot"
        }

        cars.append(car)
        seen_urls.add(listing_url)

        if vin:
            seen_vins.add(vin)

        added += 1

        time.sleep(2)

    return added


# MAIN PROGRAM
load_existing_data()

for make in selected_models:

    if len(cars) >= TARGET_VEHICLES:
        break

    print("\n============================")
    print("SCRAPING:", make)
    print("============================")

    previous_page_urls = None

    for page_number in range(MAX_PAGES):

        if len(cars) >= TARGET_VEHICLES:
            break

        try:
            response = requests.get(
                BASE_URL + "/search",
                params={
                    "make": make,
                    "page": page_number
                },
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                print(
                    "Page", page_number,
                    "status:", response.status_code
                )
                break

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            page_urls = set()

            for link in soup.find_all(
                "a",
                class_="row-title"
            ):

                href = link.get("href")

                if href:
                    page_urls.add(href)

            if not page_urls:
                print("No listings found.")
                break

            if previous_page_urls == page_urls:
                print("Repeated page. Stopping this make.")
                break

            previous_page_urls = page_urls

            added = scrape_page(
                soup,
                make
            )

            save_data()

            print(
                "Make:", make,
                "| Page:", page_number,
                "| New vehicles:", added,
                "| Progress:",
                str(len(cars)) + "/" + str(TARGET_VEHICLES)
            )

            time.sleep(2)

        except requests.RequestException as e:
            print("Request error:", e)
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
    data.groupby(["make", "model"]).size()
)

if len(data) >= TARGET_VEHICLES:
    print("\nTarget of 500 vehicles reached!")
else:
    print(
        "\nTarget not reached.",
        "Collected:", len(data),
        "out of", TARGET_VEHICLES
    )

print("\nSaved to:", CSV_FILE)
