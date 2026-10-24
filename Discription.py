
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

# ==========================================
# SETTINGS
# ==========================================

INPUT_FILE = "seeallcars_detailed.csv"
OUTPUT_FILE = "seeallcars_detailed_with_descriptions.csv"

USER_AGENT = "AcademicVehicleResearch/1.0"
REQUEST_DELAY = 2
SAVE_EVERY = 10

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT
})

robots_cache = {}


# ==========================================
# CLEAN TEXT
# ==========================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    return re.sub(r"\s+", " ", text).strip()


# ==========================================
# CHECK ROBOTS.TXT
# ==========================================

def allowed_by_robots(url):

    parsed = urlparse(url)
    origin = parsed.scheme + "://" + parsed.netloc

    if origin not in robots_cache:

        robots_url = origin + "/robots.txt"

        try:
            response = session.get(
                robots_url,
                timeout=15
            )

            if response.status_code == 404:
                robots_cache[origin] = None

            elif response.status_code == 200:

                parser = RobotFileParser()
                parser.parse(response.text.splitlines())

                robots_cache[origin] = parser

            else:
                robots_cache[origin] = False

        except requests.RequestException:
            robots_cache[origin] = False

    parser = robots_cache[origin]

    if parser is False:
        return False

    if parser is None:
        return True

    return parser.can_fetch(USER_AGENT, url)


# ==========================================
# EXTRACT DEALER NOTES
# ==========================================

def extract_dealer_notes(soup):

    selectors = [
        "[data-testid='dealer-notes']",
        "[data-testid='vehicle-description']",
        ".dealer-notes",
        ".dealerNotes",
        "#dealer-notes",
        ".vehicle-description",
        ".listing-description",
        ".seller-description",
        "[itemprop='description']"
    ]

    for selector in selectors:

        element = soup.select_one(selector)

        if element:

            description = clean_text(
                element.get_text(" ", strip=True)
            )

            if len(description) >= 40:
                return description

    # Search for headings such as Dealer Notes.
    for heading in soup.find_all(
        ["h2", "h3", "h4", "h5"]
    ):

        heading_text = clean_text(
            heading.get_text(" ", strip=True)
        ).lower().rstrip(":")

        if heading_text not in [
            "dealer notes",
            "dealer comments",
            "seller comments",
            "vehicle description",
            "description"
        ]:
            continue

        parts = []

        for sibling in heading.next_siblings:

            if getattr(sibling, "name", None) in [
                "h1", "h2", "h3", "h4", "h5"
            ]:
                break

            if isinstance(sibling, str):

                text = clean_text(sibling)

            elif hasattr(sibling, "get_text"):

                if sibling.find(
                    ["h1", "h2", "h3", "h4", "h5"]
                ):
                    break

                text = clean_text(
                    sibling.get_text(" ", strip=True)
                )

            else:
                continue

            if text:
                parts.append(text)

        description = clean_text(" ".join(parts))

        if len(description) >= 40:
            return description

        parent = heading.parent

        if parent:

            parent_text = clean_text(
                parent.get_text(" ", strip=True)
            )

            if parent_text.lower().startswith(
                heading_text
            ):

                description = clean_text(
                    parent_text[len(heading_text):]
                )

                if len(description) >= 40:
                    return description

    return ""


# ==========================================
# JSON-LD DESCRIPTION FALLBACK
# ==========================================

def extract_json_description(obj):

    if isinstance(obj, list):

        for item in obj:

            result = extract_json_description(item)

            if result:
                return result

    elif isinstance(obj, dict):

        item_type = obj.get("@type", "")

        if isinstance(item_type, list):
            item_type = " ".join(
                map(str, item_type)
            )

        item_type = str(item_type).lower()

        if any(word in item_type for word in [
            "vehicle", "car", "product"
        ]):

            description = clean_text(
                obj.get("description", "")
            )

            if len(description) >= 40:
                return description

        for value in obj.values():

            if isinstance(value, (list, dict)):

                result = extract_json_description(value)

                if result:
                    return result

    return ""


def extract_description(soup):

    description = extract_dealer_notes(soup)

    if description:
        return description

    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    for script in scripts:

        try:

            obj = json.loads(
                script.string or script.get_text()
            )

            description = extract_json_description(obj)

            if description:
                return description

        except (ValueError, TypeError):
            continue

    return ""


# ==========================================
# DESCRIPTION-BASED FEATURES
# ==========================================

def extract_features(description):

    text = description.lower()

    well_maintained = bool(re.search(
        r"\b(well[- ]maintained|"
        r"meticulously maintained|"
        r"regularly serviced|"
        r"excellent maintenance)\b",
        text
    ))

    backup_camera = bool(re.search(
        r"\b(back[- ]?up camera|"
        r"rear[- ]view camera|"
        r"rearview camera|"
        r"reversing camera)\b",
        text
    ))

    return (
        int(well_maintained),
        int(backup_camera)
    )


# ==========================================
# FETCH DESCRIPTION
# ==========================================

def fetch_description(url):

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return "", "invalid_url"

    if not allowed_by_robots(url):
        return "", "robots_disallowed"

    try:

        response = session.get(
            url,
            timeout=30
        )

        if response.status_code in [403, 429]:
            return "", "blocked"

        if response.status_code != 200:
            return "", "http_" + str(
                response.status_code
            )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        description = extract_description(soup)

        if description:
            return description, "found"

        return "", "not_found"

    except requests.RequestException:
        return "", "request_error"


# ==========================================
# LOAD EXISTING DATA
# ==========================================

print("===== LOADING EXISTING VEHICLES =====")

if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(INPUT_FILE)

original = pd.read_csv(
    INPUT_FILE,
    dtype=str
).fillna("")

if "listing_url" not in original.columns:
    raise ValueError(
        "listing_url column is missing"
    )

# Resume saved progress if available.
if Path(OUTPUT_FILE).exists():

    saved = pd.read_csv(
        OUTPUT_FILE,
        dtype=str
    ).fillna("")

    if (
        len(saved) == len(original)
        and saved["listing_url"].tolist()
        == original["listing_url"].tolist()
    ):

        data = saved.copy()

        print("Resuming previous results.")

    else:

        raise ValueError(
            "Saved output does not match "
            "the original listings."
        )

else:

    data = original.copy()


# ==========================================
# CREATE TEXT COLUMNS
# ==========================================

for column in [
    "description",
    "description_status"
]:

    if column not in data.columns:
        data[column] = ""

    data[column] = data[column].fillna("").astype(str)


# ==========================================
# CREATE NUMERIC COLUMNS
# ==========================================

for column in [
    "well_maintained",
    "backup_camera"
]:

    if column not in data.columns:
        data[column] = 0

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    ).fillna(0).astype(int)


print("Existing vehicles:", len(data))


# ==========================================
# PROCESS EXISTING URLS
# ==========================================

print("\n===== COLLECTING DEALER NOTES =====")

processed = 0
blocked_hosts = set()

for index, row in data.iterrows():

    previous_status = clean_text(
        row["description_status"]
    )

    # Skip listings already checked.
    if previous_status:
        continue

    url = clean_text(
        row["listing_url"]
    )

    if not url:

        data.at[
            index, "description_status"
        ] = "missing_url"

        continue

    hostname = urlparse(url).netloc

    if hostname in blocked_hosts:
        continue

    description, status = fetch_description(
        url
    )

    if status == "blocked":

        print(
            "Access blocked for:",
            hostname
        )

        blocked_hosts.add(hostname)

    data.at[index, "description"] = description
    data.at[index, "description_status"] = status

    if status == "found":

        well_maintained, backup_camera = (
            extract_features(description)
        )

        data.at[
            index, "well_maintained"
        ] = well_maintained

        data.at[
            index, "backup_camera"
        ] = backup_camera

    processed += 1

    if processed % SAVE_EVERY == 0:

        data.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            "Checked:", processed,
            "| Descriptions found:",
            (data["description_status"] == "found").sum()
        )

    if status not in [
        "robots_disallowed",
        "invalid_url"
    ]:
        time.sleep(REQUEST_DELAY)


# ==========================================
# SAVE FINAL RESULTS
# ==========================================

data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n===== FINAL RESULTS =====")

print("Original vehicles:", len(original))
print("Output vehicles:", len(data))

print(
    "Descriptions found:",
    (data["description_status"] == "found").sum()
)

print(
    "Well-maintained mentions:",
    pd.to_numeric(
        data["well_maintained"],
        errors="coerce"
    ).fillna(0).sum()
)

print(
    "Backup-camera mentions:",
    pd.to_numeric(
        data["backup_camera"],
        errors="coerce"
    ).fillna(0).sum()
)

print("\nDescription statuses:")

print(
    data["description_status"].value_counts()
)

print("\n===== SAMPLE DESCRIPTIONS =====")

samples = data[
    data["description_status"] == "found"
]["description"].head(3)

for number, description in enumerate(
    samples, 1
):

    print(
        "\nDescription", number,
        ":", description[:350]
    )

print("\nSaved:", OUTPUT_FILE)

print("\n===== COMPLETED =====")
