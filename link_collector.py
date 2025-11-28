#!/usr/bin/env python3
"""
Archive link collector

Recursively gathers all sublinks reachable from a given START URL (e.g., a
Wayback Machine snapshot) and writes the unique list of URLs to a JSON file.
By default, the crawl stays on the same domain as START; pass --all-domains to
follow every discovered link.

The collector keeps only the most recent Wayback snapshot for each original
page so you do not get duplicates of the same page with older timestamps.

Example
-------
python link_collector.py --start https://web.archive.org/web/20250915014250/https://www.makeitfrom.com/ --out links.json
"""

import argparse
import contextlib
import json
import time
import urllib.parse
from queue import Empty, Queue
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

DEFAULT_UA = "ArchiveLinkCollector/1.0 (+contact: your_email@example.com)"


def canonicalize(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    parts = parts._replace(fragment="")
    return urllib.parse.urlunsplit(parts)


def parse_wayback(url: str) -> Optional[tuple]:
    """Return (timestamp_int, original_url) if this is a Wayback snapshot link."""
    parts = urllib.parse.urlsplit(url)
    if parts.netloc != "web.archive.org":
        return None

    path = parts.path
    # Format: /web/<timestamp>[id_]/<original>
    if not path.startswith("/web/"):
        return None

    remainder = path[len("/web/"):]
    ts_and_rest = remainder.split("/", 1)
    if len(ts_and_rest) != 2:
        return None
    ts_raw, orig_path = ts_and_rest
    ts_digits = "".join(ch for ch in ts_raw if ch.isdigit())
    if len(ts_digits) != 14:
        return None

    try:
        ts_val = int(ts_digits)
    except ValueError:
        return None

    # Reconstruct the original URL using the parsed parts
    orig_full = orig_path
    # If the original URL already has a scheme, keep it; otherwise use https.
    if not orig_full.startswith(("http://", "https://")):
        orig_full = "https://" + orig_full

    return ts_val, canonicalize(orig_full)


def normalize_href(base_url: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = href.strip()
    if not href or href.startswith("#"):
        return None

    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/web/"):
        href = urllib.parse.urljoin("https://web.archive.org", href)
    elif href.startswith("/"):
        href = urllib.parse.urljoin(base_url, href)
    elif not href.lower().startswith("http"):
        href = urllib.parse.urljoin(base_url, href)

    return canonicalize(href)


def crawl_links(
    start: str,
    out_path: str,
    delay: float,
    user_agent: str,
    same_domain: bool = True,
    workers: int = 8,
) -> None:
    start_netloc = urllib.parse.urlsplit(start).netloc
    wb_info = parse_wayback(start)
    if wb_info:
        _, orig_url = wb_info
        start_netloc = urllib.parse.urlsplit(orig_url).netloc

    queue: Queue[str] = Queue()
    in_queue = set()
    fetched = set()
    best_snapshots = {}
    plain_links = set()

    try:
        import threading
    except Exception:
        threading = None  # type: ignore

    lock = threading.Lock() if threading else contextlib.nullcontext()

    def synchronized(fn):
        if lock is None:
            return fn

        def wrapper(*args, **kwargs):
            with lock:
                return fn(*args, **kwargs)

        return wrapper

    @synchronized
    def maybe_enqueue(url: str) -> None:
        nonlocal start_netloc
        parsed = urllib.parse.urlsplit(url)
        if same_domain:
            wb = parse_wayback(url)
            candidate_netloc = urllib.parse.urlsplit(wb[1]).netloc if wb else parsed.netloc
            if candidate_netloc != start_netloc:
                return

        wb = parse_wayback(url)
        if wb:
            ts_val, orig_url = wb
            current = best_snapshots.get(orig_url)
            if current and ts_val <= current[0]:
                return
            best_snapshots[orig_url] = (ts_val, canonicalize(url))
            if url not in fetched and url not in in_queue:
                queue.put(url)
                in_queue.add(url)
            return

        canon = canonicalize(url)
        if canon in plain_links:
            return
        plain_links.add(canon)
        if canon not in fetched and canon not in in_queue:
            queue.put(canon)
            in_queue.add(canon)

    maybe_enqueue(start)

    pbar = tqdm(total=0, unit="url", dynamic_ncols=True)
    stop_event = threading.Event() if threading else None

    def session_factory():
        s = requests.Session()
        s.headers.update({"User-Agent": user_agent})
        return s

    def worker_loop():
        session = session_factory()
        while True:
            try:
                current = queue.get(timeout=1.0)
            except Empty:
                if stop_event is None:
                    return
                if stop_event.is_set():
                    return
                continue

            if current is None:
                queue.task_done()
                return

            with lock:
                in_queue.discard(current)
                if current in fetched:
                    queue.task_done()
                    continue
                fetched.add(current)
                queued_now = queue.qsize()
                discovered_now = len(best_snapshots) + len(plain_links)

            pbar.update(1)
            pbar.set_description(
                f"Fetched={len(fetched)} queued={queued_now} discovered={discovered_now}"
            )

            try:
                resp = session.get(current, allow_redirects=True, timeout=30)
                resp.raise_for_status()
            except Exception:
                queue.task_done()
                continue

            final_url = canonicalize(resp.url)
            maybe_enqueue(final_url)
            soup = BeautifulSoup(resp.text, "lxml")

            for a in soup.find_all("a", href=True):
                nxt = normalize_href(final_url, a["href"])
                if not nxt:
                    continue
                maybe_enqueue(nxt)

            queue.task_done()

            if delay > 0:
                time.sleep(delay)

    if threading:
        threads = []
        for _ in range(max(1, workers)):
            t = threading.Thread(target=worker_loop, daemon=True)
            threads.append(t)
            t.start()

        # Wait for the queue to drain, then signal threads to stop and join them.
        queue.join()
        if stop_event:
            stop_event.set()
        for _ in threads:
            queue.put(None)
        for t in threads:
            t.join()
    else:
        while not queue.empty():
            worker_loop()
    pbar.close()

    final_links = [snap for _, snap in best_snapshots.values()]
    final_links.extend(sorted(plain_links))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"start": start, "links": sorted(final_links)}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Collect all sublinks reachable from a START URL and save them to JSON.")
    parser.add_argument("--start", required=True, help="Starting URL (e.g., a Wayback Machine snapshot).")
    parser.add_argument("--out", default="links.json", help="Output JSON file (default: links.json).")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between requests in seconds.")
    parser.add_argument("--user-agent", default=DEFAULT_UA, help="Custom User-Agent header.")
    parser.add_argument(
        "--all-domains",
        action="store_true",
        help="If set, follow links to any domain instead of staying on the START domain.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads to fetch links (default: 8).",
    )
    args = parser.parse_args()

    crawl_links(
        start=args.start,
        out_path=args.out,
        delay=args.delay,
        user_agent=args.user_agent,
        same_domain=not args.all_domains,
        workers=max(1, args.workers),
    )


if __name__ == "__main__":
    main()
