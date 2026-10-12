"""Fetch FinanceBench files and the EDGAR distractor corpus. Idempotent and resumable.

FinanceBench is fetched over HTTPS (GitHub API tree + raw files), no git needed.
Existing files with the right size are skipped. EDGAR HTML is cached on disk.

Usage:
    python scripts/download_data.py [--skip-fb] [--skip-edgar]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from citesec.config import REPO_ROOT, load_config
from citesec.ingest import edgar as ed
from citesec.ingest.financebench import load_docs

log = logging.getLogger("download")
WANTED_DATA = {
    "data/financebench_open_source.jsonl",
    "data/financebench_document_information.jsonl",
}


def _wanted(path: str) -> bool:
    return path in WANTED_DATA or (path.startswith("pdfs/") and path.endswith(".pdf"))


def fetch_financebench(cfg) -> None:
    fcfg = cfg["financebench"]
    repo, branch = fcfg["repo"], fcfg["branch"]
    api = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    raw = f"https://raw.githubusercontent.com/{repo}/{branch}/"
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        tree = client.get(api)
        tree.raise_for_status()
        entries = [
            e
            for e in tree.json()["tree"]
            if e["type"] == "blob"
            and _wanted(e["path"])
        ]
        log.info("FinanceBench: %d files wanted", len(entries))
        for e in tqdm(entries, desc="financebench"):
            dest = cfg.fb_dir / e["path"]
            if dest.exists() and dest.stat().st_size == e["size"]:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with client.stream("GET", raw + e["path"]) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            tmp.replace(dest)


def fetch_edgar(cfg) -> None:
    load_dotenv(REPO_ROOT / ".env")
    identity = os.environ.get("EDGAR_IDENTITY", "").strip()
    if "@" not in identity:
        sys.exit('Set EDGAR_IDENTITY="Your Name your.email@domain.com" in .env (SEC requires it).')
    from edgar import set_identity

    set_identity(identity)
    ecfg = cfg["edgar"]
    fb_companies = {d.company for d in load_docs(cfg.fb_dir).values()}
    holdout = set(ecfg["holdout_tickers"])
    manifest_path = cfg.edgar_dir / "manifest.csv"
    records = {r.doc_name: r for r in ed.read_manifest(manifest_path)}
    done_tickers = {r.ticker for r in records.values()}
    throttle = ed.Throttle(ecfg["rate_limit_per_sec"])
    chosen = len(done_tickers)
    q_left = ecfg["n_10q_total"] - sum(1 for r in records.values() if r.form == "10-Q")

    for ticker in ecfg["candidates"]:
        if chosen >= ecfg["n_companies"]:
            break
        if ticker in holdout or ticker in done_tickers:
            continue
        n_q = 1 if q_left > 0 else 0
        try:
            recs = ed.download_company(ticker, cfg, fb_companies, throttle, n_10q=n_q)
        except Exception as exc:
            log.error("%s failed: %s (rerun to resume)", ticker, exc)
            continue
        if not recs:
            continue
        q_left -= sum(1 for r in recs if r.form == "10-Q")
        for r in recs:
            records[r.doc_name] = r
        chosen += 1
        ed.write_manifest(list(records.values()), manifest_path)  # checkpoint per company
        log.info("%s: %d filings (%d/%d companies)", ticker, len(recs), chosen, ecfg["n_companies"])
    log.info("EDGAR manifest: %d filings, %d companies", len(records), chosen)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fb", action="store_true")
    ap.add_argument("--skip-edgar", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if not args.skip_fb:
        fetch_financebench(cfg)
    if not args.skip_edgar:
        fetch_edgar(cfg)


if __name__ == "__main__":
    main()
