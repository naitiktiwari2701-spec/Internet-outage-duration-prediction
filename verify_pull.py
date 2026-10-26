"""Check that outages_raw.csv is a COMPLETE pull. Run locally with your token set:
   export CLOUDFLARE_API_TOKEN="..."   then   python3 verify_pull.py
It requests each year separately and pages with offset, then compares to your 866 rows."""
import os, requests, pandas as pd
from datetime import datetime, timezone
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
H = {"Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}"}
URL = "https://api.cloudflare.com/client/v4/radar/annotations/outages"
have = set(int(i) for i in pd.read_csv("outages_raw.csv")["id"])
got = {}
for y in range(2022, 2027):
    off = 0
    while True:
        p = {"dateStart": f"{y}-01-01T00:00:00Z", "dateEnd": min(f"{y+1}-01-01T00:00:00Z", NOW),
             "limit": 100, "offset": off, "format": "json"}
        r = requests.get(URL, headers=H, params=p); r.raise_for_status()
        ann = r.json()["result"]["annotations"]
        for a in ann: got[int(a["id"])] = a
        print(y, "offset", off, "->", len(ann))
        if len(ann) < 100: break
        off += 100
print("\nTotal unique ids fetched:", len(got), "| in your file:", len(have))
print("In API but NOT in your file:", sorted(set(got) - have)[:30])
print("In your file but NOT in API:", sorted(have - set(got))[:30])

# Save the complete pull in the same format as outages_raw.csv so we can rebuild the clean dataset from it
pd.DataFrame(sorted(got.values(), key=lambda a: -int(a["id"]))).to_csv("outages_raw_v2.csv", index=False)
print("\nSaved", len(got), "events to outages_raw_v2.csv")
