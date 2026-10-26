import requests
import pandas as pd
import os

TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "PASTE_YOUR_TOKEN_HERE")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Pull the last ~3 years of outage annotations (adjust dateStart as needed)
url = "https://api.cloudflare.com/client/v4/radar/annotations/outages"
params = {
    "dateStart": "2022-01-01T00:00:00Z",
    "dateEnd": "2026-09-01T00:00:00Z",
    "limit": 1000,
    "format": "json"
}

resp = requests.get(url, headers=HEADERS, params=params)
resp.raise_for_status()
data = resp.json()

annotations = data["result"]["annotations"]
df = pd.DataFrame(annotations)

print(f"Pulled {len(df)} outage records")
print(df.head(10))
print("\nOutage cause counts:")
print(df["outage"].apply(lambda x: x.get("outageCause") if isinstance(x, dict) else None).value_counts())

df.to_csv("outages_raw.csv", index=False)
print("\nSaved to outages_raw.csv")
