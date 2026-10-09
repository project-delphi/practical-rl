#!/usr/bin/env python3
"""Look up on-demand prices for the workshop's AWS GPU options in the public AWS Price List.

Standard library only, and no AWS credentials: the Price List bulk files are public.

What it does:
  1. Reads the offer index (OFFER_INDEX) to find the current EC2 and SageMaker region files.
  2. Streams the region's EC2 offer CSV row by row (about 300 MB for us-east-1). Nothing is
     saved to disk and the file is never held in memory.
  3. Keeps the on-demand, Linux, shared-tenancy, no-pre-installed-software, CapacityStatus=Used
     rows for INSTANCE_TYPES, plus the EBS gp3 storage price.
  4. Streams the SageMaker offer CSV (about 1.5 MB) for ml.g4dn.xlarge in a Studio JupyterLab
     space and on a notebook instance.
  5. Writes prices.json next to this file and re-renders the generated cost table in README.md.

Usage:
  python infra/aws/price_table.py                    # us-east-1
  python infra/aws/price_table.py --region eu-west-1
  python infra/aws/price_table.py --render-only      # rebuild the README table from prices.json

Run it by hand (or from a scheduled job), never at site render time. Every price it writes
carries the offer's publication date, its version and the source URL.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRICES_JSON = HERE / "prices.json"
README = HERE / "README.md"

PRICING_HOST = "https://pricing.us-east-1.amazonaws.com"
OFFER_INDEX = PRICING_HOST + "/offers/v1.0/aws/index.json"

INSTANCE_TYPES = ("g4dn.xlarge", "g5.xlarge", "g6.xlarge")
SAGEMAKER_INSTANCE = "ml.g4dn.xlarge"
# usageType suffixes in the AmazonSageMaker offer. The prefix is a region code such as "USE1-".
SAGEMAKER_USAGE = {
    "studio_jupyterlab": "Studio:JupyterLab-" + SAGEMAKER_INSTANCE,
    "notebook_instance": "Notebk:" + SAGEMAKER_INSTANCE,
}
EBS_VOLUME_API_NAME = "gp3"

# Illustrative usage for the README: 5 workshop days x 8 running hours, and the default volume.
EXAMPLE_HOURS = 40
EXAMPLE_VOLUME_GIB = 100

EC2_FILTERS = {
    "TermType": "OnDemand",
    "Product Family": "Compute Instance",
    "Location Type": "AWS Region",
    "Operating System": "Linux",
    "Tenancy": "Shared",
    "Pre Installed S/W": "NA",
    "CapacityStatus": "Used",
    "License Model": "No License required",
    "operation": "RunInstances",
    "Unit": "Hrs",
    "Currency": "USD",
}
EBS_FILTERS = {
    "TermType": "OnDemand",
    "Product Family": "Storage",
    "Location Type": "AWS Region",
    "Volume API Name": EBS_VOLUME_API_NAME,
    "Unit": "GB-Mo",
    "Currency": "USD",
}
SAGEMAKER_FILTERS = {
    "TermType": "OnDemand",
    "Location Type": "AWS Region",
    "Unit": "Hrs",
    "Currency": "USD",
}

README_BEGIN = (
    "<!-- BEGIN generated prices (python infra/aws/price_table.py; do not edit by hand) -->"
)
README_END = "<!-- END generated prices -->"


def log(msg: str) -> None:
    print(f"price_table: {msg}", file=sys.stderr, flush=True)


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def current_region_csv(offer_code: str, region: str) -> str:
    """Return the CSV URL of the current offer file for one service and region."""
    index = fetch_json(OFFER_INDEX)
    try:
        region_index_path = index["offers"][offer_code]["currentRegionIndexUrl"]
    except KeyError as exc:
        raise SystemExit(f"offer {offer_code!r} not found in {OFFER_INDEX}") from exc
    region_index = fetch_json(PRICING_HOST + region_index_path)
    try:
        version_path = region_index["regions"][region]["currentVersionUrl"]
    except KeyError as exc:
        raise SystemExit(f"region {region!r} not found for {offer_code}") from exc
    if not version_path.endswith(".json"):
        raise SystemExit(f"unexpected offer path: {version_path}")
    return PRICING_HOST + version_path[: -len(".json")] + ".csv"


class _CountingReader(io.RawIOBase):
    """Wrap an HTTP response so we can log progress while streaming."""

    def __init__(self, raw, total: int | None, label: str) -> None:
        self._raw = raw
        self._total = total
        self._label = label
        self._seen = 0
        self._next_report = 50 * 2**20

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        data = self._raw.read(len(buf))
        n = len(data)
        buf[:n] = data
        self._seen += n
        if self._seen >= self._next_report:
            total = f" of {self._total / 2**20:.0f}" if self._total else ""
            log(f"{self._label}: {self._seen / 2**20:.0f}{total} MiB read")
            self._next_report += 50 * 2**20
        return n


def stream_offer_csv(url: str, label: str, meta: dict[str, str]):
    """Yield one dict per price row of an offer CSV, without storing the file.

    The file starts with key/value metadata lines ("Publication Date", "Version", ...),
    which are copied into ``meta``, then a header row that begins with "SKU", then one
    row per price dimension.
    """
    log(f"streaming {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        length = resp.headers.get("Content-Length")
        raw = _CountingReader(resp, int(length) if length else None, label)
        text = io.TextIOWrapper(
            io.BufferedReader(raw, buffer_size=1 << 20), encoding="utf-8", newline=""
        )
        reader = csv.reader(text)
        header: list[str] | None = None
        for row in reader:
            if not row:
                continue
            if header is None:
                if row[0] == "SKU":
                    header = row
                elif len(row) >= 2:
                    meta[row[0]] = row[1]
                continue
            yield dict(zip(header, row, strict=True))
        if header is None:
            raise SystemExit(f"no header row found in {url}")


def matches(row: dict, filters: dict, region: str) -> bool:
    if row.get("Region Code") != region:
        return False
    if "MarketOption" in row and row["MarketOption"] not in ("", "OnDemand"):
        return False
    return all(row.get(k) == v for k, v in filters.items())


def one(found: dict, key: str, what: str) -> dict:
    rows = found.get(key, [])
    if len(rows) != 1:
        detail = "; ".join(r.get("PriceDescription", "?") for r in rows) or "no rows"
        raise SystemExit(f"expected exactly one price row for {what}, found {len(rows)}: {detail}")
    return rows[0]


def ec2_prices(region: str) -> dict:
    url = current_region_csv("AmazonEC2", region)
    found: dict[str, list[dict]] = {}
    meta: dict[str, str] = {}
    for row in stream_offer_csv(url, "AmazonEC2", meta):
        if row.get("Instance Type") in INSTANCE_TYPES and matches(row, EC2_FILTERS, region):
            found.setdefault(row["Instance Type"], []).append(row)
        elif row.get("Volume API Name") == EBS_VOLUME_API_NAME and matches(
            row, EBS_FILTERS, region
        ):
            found.setdefault("ebs", []).append(row)
    instances = []
    for itype in INSTANCE_TYPES:
        row = one(found, itype, f"EC2 {itype} in {region}")
        instances.append(
            {
                "instance_type": itype,
                "gpu_count": row.get("GPU", ""),
                "gpu_memory": row.get("GPU Memory", ""),
                "vcpu": row.get("vCPU", ""),
                "memory": row.get("Memory", ""),
                "usd_per_hour": float(row["PricePerUnit"]),
                "price_description": row.get("PriceDescription", ""),
                "effective_date": row.get("EffectiveDate", ""),
                "sku": row.get("SKU", ""),
            }
        )
    ebs = one(found, "ebs", f"EBS {EBS_VOLUME_API_NAME} storage in {region}")
    return {
        "offer_code": "AmazonEC2",
        "source_url": url,
        "publication_date": meta.get("Publication Date", ""),
        "version": meta.get("Version", ""),
        "filters": {**EC2_FILTERS, "Region Code": region},
        "instances": instances,
        "ebs_gp3": {
            "usd_per_gb_month": float(ebs["PricePerUnit"]),
            "price_description": ebs.get("PriceDescription", ""),
            "effective_date": ebs.get("EffectiveDate", ""),
            "sku": ebs.get("SKU", ""),
        },
    }


def sagemaker_prices(region: str) -> dict:
    url = current_region_csv("AmazonSageMaker", region)
    found: dict[str, list[dict]] = {}
    meta: dict[str, str] = {}
    for row in stream_offer_csv(url, "AmazonSageMaker", meta):
        usage = row.get("usageType", "")
        for key, suffix in SAGEMAKER_USAGE.items():
            if usage.endswith("-" + suffix) and matches(row, SAGEMAKER_FILTERS, region):
                found.setdefault(key, []).append(row)
    items = []
    for key in SAGEMAKER_USAGE:
        row = one(found, key, f"SageMaker {key} {SAGEMAKER_INSTANCE} in {region}")
        items.append(
            {
                "option": key,
                "instance_type": SAGEMAKER_INSTANCE,
                "component": row.get("Component", ""),
                "usage_type": row.get("usageType", ""),
                "gpu": row.get("Physical GPU", ""),
                "vcpu": row.get("V CPU", ""),
                "memory": row.get("Memory", ""),
                "usd_per_hour": float(row["PricePerUnit"]),
                "price_description": row.get("PriceDescription", ""),
                "effective_date": row.get("EffectiveDate", ""),
                "sku": row.get("SKU", ""),
            }
        )
    return {
        "offer_code": "AmazonSageMaker",
        "source_url": url,
        "publication_date": meta.get("Publication Date", ""),
        "version": meta.get("Version", ""),
        "filters": {
            **SAGEMAKER_FILTERS,
            "Region Code": region,
            "usageType suffixes": SAGEMAKER_USAGE,
        },
        "items": items,
    }


def money(x: float) -> str:
    """A listed unit price, with no rounding beyond the Price List's own 4 to 6 decimals."""
    return f"{x:.6f}".rstrip("0").rstrip(".")


def render_table(data: dict) -> str:
    ec2 = data["ec2"]
    sm = data.get("sagemaker")
    lines = [
        README_BEGIN,
        "",
        f"On-demand list prices in **{data['region']}**, checked **{data['checked']}** (UTC) by "
        "`price_table.py`. They exclude tax, data transfer, and any discounts or credits.",
        "",
        f"| Option | Instance type | GPUs | vCPU | Memory | USD per hour | {EXAMPLE_HOURS} running hours |",
        "|---|---|---|---|---|---|---|",
    ]
    for it in ec2["instances"]:
        lines.append(
            f"| EC2 (Option A) | `{it['instance_type']}` | {it['gpu_count']} | {it['vcpu']} | {it['memory']} | "
            f"{money(it['usd_per_hour'])} | {it['usd_per_hour'] * EXAMPLE_HOURS:.2f} |"
        )
    if sm:
        names = {
            "studio_jupyterlab": "SageMaker Studio JupyterLab space (Option B)",
            "notebook_instance": "SageMaker notebook instance",
        }
        for it in sm["items"]:
            lines.append(
                f"| {names.get(it['option'], it['option'])} | `{it['instance_type']}` | 1 | {it['vcpu']} | "
                f"{it['memory']} | {money(it['usd_per_hour'])} | {it['usd_per_hour'] * EXAMPLE_HOURS:.2f} |"
            )
    gp3 = ec2["ebs_gp3"]["usd_per_gb_month"]
    lines += [
        "",
        f"EBS gp3 storage: {money(gp3)} USD per GB-month, so the default {EXAMPLE_VOLUME_GIB} GiB volume "
        f"costs about {gp3 * EXAMPLE_VOLUME_GIB:.2f} USD per month, billed while the volume exists "
        "(running or stopped) and prorated.",
        "",
        f"The last column is the hourly price × {EXAMPLE_HOURS} hours (5 days × 8 hours), "
        "an illustration rather than a measurement.",
        "",
        "Sources:",
        f"- EC2: <{ec2['source_url']}> (publication date {ec2['publication_date']}, version {ec2['version']})",
    ]
    if sm:
        lines.append(
            f"- SageMaker: <{sm['source_url']}> (publication date {sm['publication_date']}, version {sm['version']})"
        )
    lines += ["", README_END]
    return "\n".join(lines)


def update_readme(data: dict) -> None:
    if not README.exists():
        log(f"{README} not found; skipping the README table")
        return
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_BEGIN) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        log("README has no generated-prices markers; skipping the README table")
        return
    README.write_text(pattern.sub(lambda _: render_table(data), text), encoding="utf-8")
    log(f"updated the cost table in {README}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--out", type=Path, default=PRICES_JSON)
    parser.add_argument("--no-sagemaker", action="store_true", help="skip the SageMaker offer")
    parser.add_argument("--render-only", action="store_true", help="re-render README.md from --out")
    args = parser.parse_args(argv)

    if args.render_only:
        update_readme(json.loads(args.out.read_text(encoding="utf-8")))
        return 0

    data = {
        "checked": dt.datetime.now(dt.UTC).date().isoformat(),
        "region": args.region,
        "currency": "USD",
        "generator": "infra/aws/price_table.py",
        "note": "On-demand list prices from the public AWS Price List bulk files. "
        "They exclude tax, data transfer, and any discounts or credits.",
        "ec2": ec2_prices(args.region),
        "sagemaker": None if args.no_sagemaker else sagemaker_prices(args.region),
    }
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {args.out}")
    for it in data["ec2"]["instances"]:
        log(f"{it['instance_type']}: {it['usd_per_hour']} USD/hour")
    if data["sagemaker"]:
        for it in data["sagemaker"]["items"]:
            log(f"{it['option']} {it['instance_type']}: {it['usd_per_hour']} USD/hour")
    if args.out == PRICES_JSON:
        update_readme(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
