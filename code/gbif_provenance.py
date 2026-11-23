"""Retrieve GBIF metadata for every GBIF image used in the study.

For each occurrence key in the auxiliary subsets and the GBIF reference set, the script queries the
public GBIF occurrence API and records the source dataset, record licence, media licence, creator and
event date. It also writes the number of records per source dataset, which is the input GBIF needs to
register a derived dataset with a citable DOI.

Inputs : gbif_membership.csv and gbif_reference_disjoint.txt (partition files of the study)
Outputs: gbif_occurrence_metadata.csv and gbif_derived_dataset_counts.csv in OUT_DIR
Usage  : python gbif_provenance.py PARTITION_DIR OUT_DIR
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

PART, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
API = "https://api.gbif.org/v1/occurrence/"


def key_of(path):
    return Path(path.strip()).stem


members = pd.read_csv(PART / "gbif_membership.csv")
members["occurrence_key"] = members["image_file"].map(key_of)
reference = [key_of(x) for x in (PART / "gbif_reference_disjoint.txt").read_text().splitlines() if x.strip()]
keys = list(dict.fromkeys(list(members["occurrence_key"]) + reference))


def fetch(key):
    for attempt in range(4):
        try:
            req = urllib.request.Request(API + key, headers={"User-Agent": "gbif-provenance/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                o = json.loads(r.read())
            m = (o.get("media") or [{}])[0]
            return dict(occurrence_key=key, dataset_key=o.get("datasetKey"), dataset_title=o.get("datasetTitle"),
                        record_licence=o.get("license"), media_licence=m.get("license"), creator=m.get("creator"),
                        rights_holder=m.get("rightsHolder"), event_date=o.get("eventDate"), country=o.get("countryCode"),
                        media_url=m.get("identifier"))
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 * (attempt + 1))
    return dict(occurrence_key=key, error=err)


with cf.ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(fetch, keys))
meta = pd.DataFrame(rows)
role = {k: [] for k in keys}
for _, r in members.iterrows():
    for n in ("N500", "N2000", "N5000"):
        if isinstance(r[n], str) and r[n]:
            role[r.occurrence_key].append(f"{n}_{r[n]}")
for k in reference:
    role[k].append("reference_set")
meta["use"] = meta["occurrence_key"].map(lambda k: ";".join(role[k]))
meta.to_csv(OUT / "gbif_occurrence_metadata.csv", index=False)
counts = meta.groupby("dataset_key").size().rename("count").reset_index()
counts.to_csv(OUT / "gbif_derived_dataset_counts.csv", index=False)
ok = meta[meta.get("error").isna()] if "error" in meta else meta
print("records:", len(meta), "| errors:", int(meta["error"].notna().sum()) if "error" in meta else 0,
      "| datasets:", len(counts), "| latest event date:", ok["event_date"].dropna().astype(str).str[:10].max())
print(ok["media_licence"].value_counts().to_string())
