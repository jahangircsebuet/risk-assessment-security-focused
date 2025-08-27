#!/usr/bin/env python3
import argparse, json, os, re, sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import zstandard as zstd

URL_RE = re.compile(r"https?://\S+")

def iter_jsonl_zst(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while True:
                chunk = reader.read(2**20)
                if not chunk:
                    break
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl == -1:
                        break
                    line = buf[:nl]
                    buf = buf[nl+1:]
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

def normalize_submission(obj):
    author = obj.get("author") or "[deleted]"
    title = obj.get("title") or ""
    selftext = obj.get("selftext") or ""
    text = (title + " " + selftext).strip()
    ts = pd.to_datetime(obj.get("created_utc", 0), unit="s", utc=True)
    url = obj.get("url") or ""
    has_media = bool(url or URL_RE.search(text))
    return dict(
        user_id=str(author),
        timestamp=ts,
        text=text,
        audience="public",
        geotag=False,
        has_media=has_media,
        source_kind="submission",
        subreddit=obj.get("subreddit") or "",
        id=str(obj.get("id") or ""),
        url=url
    )

def normalize_comment(obj):
    author = obj.get("author") or "[deleted]"
    body = obj.get("body") or ""
    ts = pd.to_datetime(obj.get("created_utc", 0), unit="s", utc=True)
    has_media = bool(URL_RE.search(body))
    return dict(
        user_id=str(author),
        timestamp=ts,
        text=body,
        audience="public",
        geotag=False,
        has_media=has_media,
        source_kind="comment",
        subreddit=obj.get("subreddit") or "",
        id=str(obj.get("id") or ""),
        url=""  # comments have permalink but skipping here
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input .zst file (RS_*.zst or RC_*.zst)")
    ap.add_argument("--out_dir", required=True, help="Output directory for Parquet")
    ap.add_argument("--subreddits", nargs="*", default=None, help="Optional whitelist (e.g., askscience science)")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N rows (debug)")
    args = ap.parse_args()

    inp = Path(args.inp)
    kind = inp.name.split("_")[0]  # RS or RC
    rows = []
    for i, obj in enumerate(iter_jsonl_zst(inp)):
        if kind == "RS":
            row = normalize_submission(obj)
        else:
            row = normalize_comment(obj)

        if args.subreddits and row["subreddit"].lower() not in {s.lower() for s in args.subreddits}:
            continue

        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        print("No rows parsed.", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = inp.stem.replace(".zst","")
    out_path = out_dir / f"{base}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows.")

if __name__ == "__main__":
    main()
