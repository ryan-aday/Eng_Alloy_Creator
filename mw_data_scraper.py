#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Engineering ToolBox Crawler & Ingestor (RAG-ready)

- Discovers & crawls ET article/category pages
- Extracts: title, cleaned text blocks, "This page can be cited as ..." citation,
  tables (to CSV), images (downloaded), equations, "Where:" symbols, outlinks
- Outputs:
    out/
      corpus.jsonl              # one record per page
      corpus_chunks.jsonl       # (optional) chunked RAG segments
      img/                      # downloaded images
      tables/                   # CSVs from HTML tables
      crawl_state.json          # resume state
- No global UA scoping issues: user agent is passed explicitly

USAGE EXAMPLES
--------------
# Minimal crawl (default seeds), polite delay:
python mw_data_scraper.py --out ./mw_corpus

# Resume later:
python mw_data_scraper.py --out ./mw_corpus --resume

# Faster (only if you have permission), chunk for RAG:
python mw_data_scraper.py --out ./mw_corpus --delay 0.25 --chunk

# Limit page count for testing:
python mw_data_scraper.py --out ./mw_corpus --max-pages 100

# Re-seed using URLs/outlinks from a prior run's corpus:
python mw_data_scraper.py --out ./mw_corpus_v2 --from-corpus ./mw_corpus/corpus.jsonl --chunk
"""

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from urllib import robotparser

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ------------------------
# Constants & patterns
# ------------------------
ROOT = "https://web.archive.org/web/20250915014250/https://www.makeitfrom.com/"
ORIGINAL_ROOT = "https://www.makeitfrom.com/"
ARCHIVE_HOST = "https://web.archive.org"
ARCHIVE_RE = re.compile(r"^(https?://web\.archive\.org/web/[^/]+/)(.*)$", re.IGNORECASE)

ARTICLE_PAT = re.compile(r"-[dt]_\d+\.html$", re.IGNORECASE)
HTML_PAT = re.compile(r"\.html?$", re.IGNORECASE)
EXCLUDE_PAT = re.compile(r"\.(pdf|zip|rar|7z|exe|docx?)$", re.IGNORECASE)

DEFAULT_UA = "ET-RAG-Crawler/1.1 (+contact: your_email@example.com)"

# Greek & sub/superscript normalization
SUB_MAP = str.maketrans("0123456789+-=()n", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙ")
SUP_MAP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
GREEK = {
    "&alpha;": "α", "&beta;": "β", "&gamma;": "γ", "&delta;": "δ", "&epsilon;": "ε",
    "&theta;": "θ", "&lambda;": "λ", "&mu;": "μ", "&nu;": "ν", "&pi;": "π", "&rho;": "ρ",
    "&sigma;": "σ", "&tau;": "τ", "&phi;": "φ", "&omega;": "ω",
}

# Equation & symbol patterns
EQ_LINE = re.compile(
    r"\b([A-Za-zΑ-Ωα-ωσ][A-Za-z0-9_α-ωΑ-Ωσ]*\s*(?:₀|₁|₂|₃|₄|₅|₆|₇|₈|₉|[0-9]*)?\s*"
    r"(?:=|≈|≃|≅|≤|≥|∝)\s*[^;\n\r]+)"
)
WHERE_HEAD = re.compile(r"^\s*(?:where|Where|Symbols)\s*[:：]?\s*$")
SYM_LINE = re.compile(
    r"^\s*([A-Za-zΑ-Ωα-ωσ][A-Za-z0-9_α-ωΑ-Ωσ]*(?:\s*(?:₀|₁|₂|₃|₄|₅|₆|₇|₈|₉|[0-9])*)?)\s*[-–:=]\s*(.+?)\s*$"
)

# ------------------------
# Utilities
# ------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_filename(s: str, maxlen: int = 220) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return s[:maxlen]

def canonicalize_url(url: str) -> str:
    u = urllib.parse.urlsplit(url)
    u = u._replace(fragment="")
    return urllib.parse.urlunsplit(u)

def _split_archive_url(url: str) -> Tuple[Optional[str], str]:
    """Return (archive_prefix, original_url) for a web.archive.org URL.

    If the URL is not an archive URL, archive_prefix is None and original_url is the input.
    """
    m = ARCHIVE_RE.match(url)
    if not m:
        return None, url
    return m.group(1), m.group(2)

def _wrap_archive(original_url: str, archive_prefix: Optional[str]) -> str:
    if archive_prefix:
        return f"{archive_prefix}{original_url}"
    return original_url

def is_makeitfrom_url(url: str) -> bool:
    """Return True if the URL points to the MakeItFrom domain (archived or live)."""
    _, original = _split_archive_url(url)
    return original.startswith(ORIGINAL_ROOT)

def join_url(base: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = href.strip()
    if href.startswith("#"):
        return None
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("/web/"):
        # Already an archive path; join directly against archive host to avoid double-wrapping.
        href = urllib.parse.urljoin(ARCHIVE_HOST, href)

    archive_prefix, base_original = _split_archive_url(base)

    if href.startswith("http"):
        href_archive_prefix, href_original = _split_archive_url(href)
        if href_archive_prefix:
            archive_prefix = href_archive_prefix
            target_original = canonicalize_url(href_original)
        else:
            target_original = canonicalize_url(href)
    else:
        # Join relative URLs against the original (non-archive) location, then wrap back.
        target_original = canonicalize_url(urllib.parse.urljoin(base_original, href))

    # Preserve archive prefix if we started from an archived page.
    target = _wrap_archive(target_original, archive_prefix)
    return canonicalize_url(target)

def is_article_like(url: str) -> bool:
    _, original = _split_archive_url(url)
    if not original.startswith(ORIGINAL_ROOT):
        return False

    if EXCLUDE_PAT.search(url):
        return False
    if ARTICLE_PAT.search(url):
        return True
    if HTML_PAT.search(original):
        return True
    return False

def read_robots_txt(ignore: bool, ua: str) -> Optional[robotparser.RobotFileParser]:
    if ignore:
        return None
    rp = robotparser.RobotFileParser()
    robots_url = urllib.parse.urljoin(ORIGINAL_ROOT, "robots.txt")
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp
    except Exception:
        return None

def allowed_by_robots(rp: Optional[robotparser.RobotFileParser], ua: str, url: str) -> bool:
    if rp is None:
        return True
    try:
        _, original = _split_archive_url(url)
        return rp.can_fetch(ua, original)
    except Exception:
        return True

def fetch(url: str, ua: str, timeout: float = 30.0, retries: int = 3, backoff: float = 1.5) -> Optional[requests.Response]:
    last_exc = None
    for i in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": ua}, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r
            if r.status_code >= 500 or not r.content:
                time.sleep(backoff ** (i + 1))
            else:
                return None
        except Exception as e:
            last_exc = e
            time.sleep(backoff ** (i + 1))
    return None

# ------------------------
# Content extraction
# ------------------------
def _html_entity_to_unicode(text: str) -> str:
    if not text:
        return text
    for k, v in GREEK.items():
        text = text.replace(k, v)
    return html.unescape(text)

def _flatten_math_tags(el) -> str:
    """Convert <sub>/<sup> to unicode subs/supers, preserve newlines, decode entities."""
    parts = []
    for node in el.descendants:
        name = getattr(node, "name", None)
        if name == "sub":
            parts.append(str(node.get_text()).translate(SUB_MAP))
        elif name == "sup":
            parts.append(str(node.get_text()).translate(SUP_MAP))
        elif name == "br":
            parts.append("\n")
        elif not name:
            parts.append(str(node))
    txt = _html_entity_to_unicode("".join(parts))
    txt = re.sub(r"[ \t]+", " ", txt)
    return txt.strip()

def extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    title = soup.find("title")
    if title and title.get_text(strip=True):
        return title.get_text(strip=True)
    h2 = soup.find("h2")
    if h2 and h2.get_text(strip=True):
        return h2.get_text(strip=True)
    return ""

def extract_citation_text(soup: BeautifulSoup) -> str:
    txt = soup.get_text("\n", strip=True)
    m = re.search(r"This page can be cited as.*", txt, re.IGNORECASE)
    if m:
        return " ".join(m.group(0).split())
    for p in soup.find_all("p"):
        t = p.get_text(" ", strip=True)
        if "This page can be cited as" in t:
            return t
    return ""

def extract_text_blocks(soup: BeautifulSoup) -> List[str]:
    """Collect headings, paragraphs, list items, and prose-like TD/TH/DIV content."""
    for tag in soup(["script","style","noscript","header","footer","nav","form","iframe"]):
        tag.decompose()

    blocks = []
    root = soup.body or soup
    for el in root.find_all(["h1","h2","h3","p","li","pre","code","td","th","div"]):
        if el.find_parent(["nav","header","footer","form"]):
            continue
        txt = _flatten_math_tags(el).strip()
        if not txt:
            continue
        if el.name in ("td","th","div") and len(txt) < 12 and "=" not in txt:
            continue
        blocks.append(txt)

    deduped = []
    prev = None
    for b in blocks:
        if b != prev:
            deduped.append(b)
        prev = b
    return deduped

def extract_tables(url: str, html_text: str) -> List[pd.DataFrame]:
    try:
        dfs = pd.read_html(html_text)
        return dfs
    except Exception:
        return []

def extract_images(base_url: str, soup: BeautifulSoup) -> List[Dict]:
    out = []
    for img in soup.find_all("img"):
        src = img.get("src", "").strip()
        alt = img.get("alt", "").strip()
        full = join_url(base_url, src)
        if not full:
            continue
        if not full.lower().startswith("http"):
            continue
        out.append({"src": full, "alt": alt})
    return out

def extract_equations_and_symbols(soup: BeautifulSoup) -> Dict[str, list]:
    """
    Returns:
        {"equations": [str], "symbols": [{"symbol": str, "meaning": str}], "raw_blocks":[str]}
    Scans paragraphs, table cells, list items, and code/pre.
    """
    equations, symbol_pairs, raw_blocks = [], [], []
    candidates = soup.select("p, li, td, th, pre, code, div")
    for el in candidates:
        if el.find_parent(["nav","header","footer","form","script","style"]):
            continue

        txt = _flatten_math_tags(el)
        if not txt:
            continue
        raw_blocks.append(txt)

        # WHERE header → parse next siblings and list items
        for line in txt.splitlines():
            if WHERE_HEAD.match(line.strip()):
                sibs = []
                ul = el.find_next_sibling(["ul","ol"])
                if ul:
                    for li in ul.select("li"):
                        s = _flatten_math_tags(li).strip()
                        if s:
                            sibs.append(s)
                nxt = el.find_next_siblings(limit=6)
                for n in nxt:
                    if n.name in ("p","li","div","td","th"):
                        s = _flatten_math_tags(n).strip()
                        if s:
                            sibs.append(s)
                for s in sibs:
                    m = SYM_LINE.match(s)
                    if m:
                        symbol_pairs.append({"symbol": m.group(1), "meaning": m.group(2)})

        # Equations from lines and whole blocks
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            m = EQ_LINE.search(line)
            if m:
                eq = m.group(1).rstrip(" .;,")
                equations.append(eq)
        if "=" in txt or "≈" in txt or "∝" in txt:
            m2 = EQ_LINE.search(txt)
            if m2:
                eq = m2.group(1).rstrip(" .;,")
                equations.append(eq)

    # Deduplicate while preserving order
    seen = set(); equations_unique = []
    for e in equations:
        if e not in seen:
            equations_unique.append(e); seen.add(e)

    seen_s = set(); sym_unique = []
    for d in symbol_pairs:
        k = d["symbol"]
        if k not in seen_s:
            sym_unique.append(d); seen_s.add(k)

    return {"equations": equations_unique, "symbols": sym_unique, "raw_blocks": raw_blocks}

def download_image(img_url: str, out_dir: str, delay: float, ua: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    r = fetch(img_url, ua=ua, timeout=30)
    if r is None:
        return None
    parsed = urllib.parse.urlsplit(img_url)
    base = safe_filename(os.path.basename(parsed.path)) or hashlib.sha1(img_url.encode()).hexdigest() + ".bin"
    local_path = os.path.join(out_dir, base)
    try:
        with open(local_path, "wb") as f:
            f.write(r.content)
        if delay > 0:
            time.sleep(delay)
        return local_path
    except Exception:
        return None

# ------------------------
# Data model
# ------------------------
@dataclass
class PageRecord:
    url: str
    title: str
    crawl_ts: str
    citation_text: str
    text_blocks: List[str]
    tables: List[Dict]            # [{"csv_path":..., "n_rows":..., "n_cols":..., "caption": ""}]
    images: List[Dict]            # [{"src":..., "alt":"", "local_path": "..."}]
    outlinks: List[str]
    http_last_modified: Optional[str] = None
    http_etag: Optional[str] = None
    equations: List[str] = None   # <-- NEW
    symbols: List[Dict] = None    # <-- NEW

# ------------------------
# Chunking
# ------------------------
def chunk_blocks(blocks: List[str], max_words: int = 600, overlap: int = 120) -> List[str]:
    """Approximate word-count chunking with overlap; preserves block boundaries."""
    chunks = []
    cur = []
    cur_len = 0
    for b in blocks:
        words = b.split()
        wl = len(words)
        if cur_len + wl <= max_words:
            cur.append(b); cur_len += wl
        else:
            if cur:
                chunks.append("\n\n".join(cur))
            if overlap > 0 and chunks:
                tail_words = " ".join(chunks[-1].split()[-overlap:])
                cur = [tail_words, b]
                cur_len = len(tail_words.split()) + wl
            else:
                cur = [b]; cur_len = wl
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

# ------------------------
# Main
# ------------------------
def main():
    """
    Engineering ToolBox full-site crawler & ingestor (RAG-ready)
    """
    parser = argparse.ArgumentParser(
        prog="mw_data_scraper.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Crawls EngineeringToolBox.com and saves pages, text, tables, images, equations, and symbol lists\n"
            "for use in Retrieval-Augmented Generation (RAG) pipelines.\n\n"
            "Typical workflow:\n"
            "  1) Crawl ET pages -> corpus.jsonl (and optionally corpus_chunks.jsonl)\n"
            "  2) Embed corpus_chunks.jsonl into a vector DB (Chroma, LanceDB, Milvus, etc.)\n"
            "  3) Use metadata (url, title, citation_text) to return precise citations in your app."
        ),
    )

    # Required
    parser.add_argument(
        "--out", required=True,
        help="Root output directory (JSONL + images + tables). Example: --out ./mw_corpus"
    )

    # General crawling controls
    parser.add_argument(
        "--delay", type=float, default=0.8,
        help="Delay between HTTP requests in seconds (default: 0.8)."
    )
    parser.add_argument(
        "--max-pages", type=int, default=0,
        help="Maximum pages to crawl (0 = no limit)."
    )
    parser.add_argument(
        "--ignore-robots", action="store_true",
        help="Ignore robots.txt (ONLY if you have explicit permission)."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous crawl_state.json if it exists."
    )

    # Crawl scope
    parser.add_argument(
        "--seeds", nargs="*", default=[
            ROOT,
        ],
        help="Seed URLs to start crawling from (space-separated)."
    )
    parser.add_argument(
        "--from-corpus",
        help="Optional path to an existing corpus.jsonl; seeds crawl queue from saved urls/outlinks."
    )

    # Storage locations
    parser.add_argument("--img", default="img", help="Subdirectory for downloaded images (default: img).")
    parser.add_argument("--tables", default="tables", help="Subdirectory for CSV tables (default: tables).")
    parser.add_argument("--user-agent", default=DEFAULT_UA, help="Custom HTTP User-Agent.")

    # RAG options
    parser.add_argument("--chunk", action="store_true",
                        help="Also write a RAG-ready 'corpus_chunks.jsonl' with text segments.")
    parser.add_argument("--chunk-max", type=int, default=600,
                        help="Approximate words per chunk (default: 600).")
    parser.add_argument("--chunk-overlap", type=int, default=120,
                        help="Approximate word overlap between chunks (default: 120).")

    # Print help if no args
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    args = parser.parse_args()

    ua = args.user_agent

    # Prepare output dirs/files
    out_root = args.out
    os.makedirs(out_root, exist_ok=True)
    pages_dir = os.path.join(out_root, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    img_dir = os.path.join(out_root, args.img)
    os.makedirs(img_dir, exist_ok=True)
    tables_dir = os.path.join(out_root, args.tables)
    os.makedirs(tables_dir, exist_ok=True)

    corpus_jsonl = os.path.join(out_root, "corpus.jsonl")
    chunks_jsonl = os.path.join(out_root, "corpus_chunks.jsonl")
    state_path = os.path.join(out_root, "crawl_state.json")

    # Load / init state
    if args.resume and os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        todo = state.get("todo", [])
        seen_original = set(state.get("seen_original", state.get("seen", [])))
    else:
        todo = [canonicalize_url(u) for u in args.seeds]
        seen_original = set()

    # Optional: seed from previous corpus
    if args.from_corpus and os.path.exists(args.from_corpus):
        seed_urls = []
        with open(args.from_corpus, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                seed_urls.append(rec.get("url", ""))
                seed_urls.extend(rec.get("outlinks", []))
        for u in sorted(set(seed_urls)):
            if not u:
                continue
            if not is_makeitfrom_url(u) or not is_article_like(u):
                continue
            _, u_original = _split_archive_url(u)
            u_original_canon = canonicalize_url(u_original)
            if u_original_canon in seen_original:
                continue
            todo.append(u)

    rp = read_robots_txt(args.ignore_robots, ua=ua)

    pages_crawled = 0
    out_f = open(corpus_jsonl, "a", encoding="utf-8")
    pbar = tqdm(total=0, unit="page", dynamic_ncols=True)

    try:
        while todo:
            url = todo.pop(0)
            _, original_url = _split_archive_url(url)
            original_canon = canonicalize_url(original_url)

            if original_canon in seen_original:
                continue
            seen_original.add(original_canon)

            if not is_makeitfrom_url(url):
                continue

            # Follow HTML pages; keep category pages to expand links
            if rp and not allowed_by_robots(rp, ua, url):
                continue

            r = fetch(url, ua=ua)
            if r is None:
                continue

            last_mod = r.headers.get("Last-Modified")
            etag = r.headers.get("ETag")

            html_text = r.text
            soup = BeautifulSoup(html_text, "lxml")

            title = extract_title(soup)
            citation = extract_citation_text(soup)
            text_blocks = extract_text_blocks(soup)
            images = extract_images(url, soup)
            tables = extract_tables(url, html_text)
            math_extracted = extract_equations_and_symbols(soup)
            equations = math_extracted["equations"]
            symbols = math_extracted["symbols"]

            # Save tables to CSV
            table_meta = []
            for i, df in enumerate(tables):
                base = f"{safe_filename(title) or 'page'}_{hashlib.sha1((url+str(i)).encode()).hexdigest()[:10]}.csv"
                csv_path = os.path.join(tables_dir, base)
                try:
                    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
                    table_meta.append({
                        "csv_path": os.path.relpath(csv_path, out_root),
                        "n_rows": int(df.shape[0]),
                        "n_cols": int(df.shape[1]),
                        "caption": ""
                    })
                except Exception:
                    continue

            # Download images
            image_meta = []
            for im in images:
                local = download_image(im["src"], out_dir=img_dir, delay=max(args.delay/4.0, 0.0), ua=ua)
                image_meta.append({
                    "src": im["src"],
                    "alt": im.get("alt", ""),
                    "local_path": os.path.relpath(local, out_root) if local else None
                })

            # Outlinks discovery
            outlinks = []
            for a in soup.find_all("a", href=True):
                target = join_url(url, a["href"])
                if not target:
                    continue
                if not is_makeitfrom_url(target):
                    continue
                _, t_original = _split_archive_url(target)
                t_original_canon = canonicalize_url(t_original)
                if EXCLUDE_PAT.search(target):
                    continue
                if t_original_canon not in seen_original and is_article_like(target):
                    todo.append(target)
                outlinks.append(target)

            rec = PageRecord(
                url=url,
                title=title,
                crawl_ts=now_iso(),
                citation_text=citation,
                text_blocks=text_blocks,
                tables=table_meta,
                images=image_meta,
                outlinks=sorted(set(outlinks)),
                http_last_modified=last_mod,
                http_etag=etag,
                equations=equations,
                symbols=symbols
            )
            out_f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            out_f.flush()

            pages_crawled += 1
            pbar.set_description(f"Crawled {pages_crawled} | todo={len(todo)}")
            pbar.update(1)

            # Rate limit
            if args.delay > 0:
                time.sleep(args.delay)

            # Persist state periodically
            if pages_crawled % 25 == 0:
                with open(state_path, "w", encoding="utf-8") as sf:
                    json.dump({"todo": todo, "seen_original": list(seen_original)}, sf, ensure_ascii=False, indent=2)

            if args.max_pages and pages_crawled >= args.max_pages:
                break

    finally:
        out_f.close()
        with open(state_path, "w", encoding="utf-8") as sf:
            json.dump({"todo": todo, "seen_original": list(seen_original)}, sf, ensure_ascii=False, indent=2)
        pbar.close()

    # Chunking pass (optional)
    if args.chunk:
        with open(corpus_jsonl, "r", encoding="utf-8") as f_in, open(chunks_jsonl, "w", encoding="utf-8") as f_out:
            for line in f_in:
                rec = json.loads(line)
                chunks = chunk_blocks(
                    rec.get("text_blocks", []),
                    max_words=args.chunk_max,
                    overlap=args.chunk_overlap
                )
                for idx, ch in enumerate(chunks):
                    out = {
                        "doc_id": hashlib.sha1(rec["url"].encode()).hexdigest(),
                        "url": rec["url"],
                        "title": rec.get("title", ""),
                        "chunk_index": idx,
                        "chunk_text": ch,
                        "citation_text": rec.get("citation_text", ""),
                        "images": rec.get("images", []),
                        "tables": rec.get("tables", []),
                        "equations": rec.get("equations", []),
                        "symbols": rec.get("symbols", []),
                        "crawl_ts": rec.get("crawl_ts", "")
                    }
                    f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\nDone. Pages crawled: {pages_crawled}")
    print(f"Corpus: {corpus_jsonl}")
    if args.chunk:
        print(f"Chunked corpus: {chunks_jsonl}")
    print(f"Images dir: {img_dir}")
    print(f"Tables dir: {tables_dir}")
    print(f"State: {state_path}")


if __name__ == "__main__":
    main()
