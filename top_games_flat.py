#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

import requests
from bs4 import BeautifulSoup

try:
    from rapidfuzz import fuzz  # type: ignore
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False


# -----------------------------
# Config
# -----------------------------

ARCHIVE_EXTS = {".7z", ".zip", ".rar"}
JUNK_EXTS = {
    ".nfo", ".txt", ".md", ".rtf", ".diz", ".sfv", ".md5", ".sha1",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    ".url", ".db", ".ini"
}

SYSTEM_PRIMARY_EXTS: Dict[str, List[str]] = {
    "PS1":      [".chd", ".pbp", ".cue", ".bin", ".iso", ".img"],
    "PS2":      [".iso", ".chd", ".bin", ".img", ".cso", ".zso", ".nrg", ".mdf"],
    "PSP":      [".iso", ".cso", ".pbp"],
    "PSVITA":   [".vpk", ".zip"],

    "GAMECUBE": [".rvz", ".gcz", ".iso", ".gcm"],
    "WII":      [".wbfs", ".rvz", ".gcz", ".iso"],
    "WIIU":     [".wua", ".wux", ".wud", ".rpx", ".wup"],
    "SWITCH":   [".xci", ".nsp", ".xcz", ".nsz"],

    "N64":      [".z64", ".n64", ".v64"],
    "NDS":      [".nds"],
    "3DS":      [".3ds", ".cia", ".cxi"],
    "GBA":      [".gba"],
    "GBC":      [".gbc", ".gb"],

    "SNES":         [".sfc", ".smc"],
    "SEGA GENESIS": [".md", ".gen", ".bin", ".smd"],
    "SEGA 32X":     [".32x", ".bin", ".md"],
    "NEO GEO":      [".chd", ".cue", ".bin", ".zip"],
    "PICO 8":       [".p8.png", ".p8"],
    "SCUMMVM":      [".scummvm"],
}

METACRITIC_URL: Dict[str, str] = {
    "PS1": "https://www.metacritic.com/browse/game/ps1/",
    "PS2": "https://www.metacritic.com/browse/game/ps2/",
    "PSP": "https://www.metacritic.com/browse/game/psp/",
    "PSVITA": "https://www.metacritic.com/browse/game/ps-vita/",
    "SWITCH": "https://www.metacritic.com/browse/game/nintendo-switch/",
    "WII": "https://www.metacritic.com/browse/game/wii/",
    "WIIU": "https://www.metacritic.com/browse/game/wii-u/",
    "GAMECUBE": "https://www.metacritic.com/browse/game/gamecube/",
    "N64": "https://www.metacritic.com/browse/game/nintendo-64/",
    "NDS": "https://www.metacritic.com/browse/game/nintendo-ds/all/all-time/metascore/",
    "3DS": "https://www.metacritic.com/browse/game/3ds/",
    "GBA": "https://www.metacritic.com/browse/game/game-boy-advance/all/all-time/",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

MONTHS_RE = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
ENTRY_RE = re.compile(
    rf"\b(?P<rank>\d+)\.\s+(?P<title>.*?)\s+{MONTHS_RE}\s+\d{{1,2}},\s+\d{{4}}.*?(?P<score>\d{{1,3}})\s+Metascore\b",
    re.DOTALL
)

BRACKET_GARBAGE_RE = re.compile(r"[\(\[\{].*?[\)\]\}]")
DISC_RE = re.compile(r"\b(disc|disk|cd)\s*\d+\b", re.IGNORECASE)
TRACK_RE = re.compile(r"\btrack\s*\d+\b", re.IGNORECASE)
PUNCT_RE = re.compile(r"[^a-z0-9\s]+")


@dataclass
class MetaEntry:
    rank: int
    title: str
    score: int


@dataclass
class Candidate:
    path: Path
    base_key: str
    label: str
    is_archive: bool


def canon_system(name: str) -> str:
    return name.strip().upper()


def safe_name(s: str, max_len: int = 160) -> str:
    s = re.sub(r'[<>:"/\\|?*]+', "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:max_len].strip()


def normalize_title(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = BRACKET_GARBAGE_RE.sub(" ", s)
    s = DISC_RE.sub(" ", s)
    s = TRACK_RE.sub(" ", s)
    s = s.replace("™", " ").replace("®", " ")
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def base_key_for_name(name: str) -> str:
    return normalize_title(name)


def polite_sleep(base: float) -> None:
    time.sleep(base + random.uniform(0.15, 0.55))


def build_page_url(base_url: str, page: int) -> str:
    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    qs["page"] = [str(page)]
    qs.setdefault("sort", ["desc"])
    new_query = urlencode(qs, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))


def fetch_text(session: requests.Session, url: str, retries: int, delay: float, timeout: int) -> str:
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code in (403, 429):
                wait = (delay * (2 ** attempt)) + random.uniform(0.5, 1.7)
                print(f"    ! HTTP {r.status_code}. Backing off {wait:.1f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            wait = (delay * (2 ** attempt)) + random.uniform(0.3, 1.0)
            print(f"    ! Fetch error: {e}. Retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch: {url} (last={last})")


def parse_metacritic(html: str) -> List[MetaEntry]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)

    got: Dict[int, MetaEntry] = {}
    for m in ENTRY_RE.finditer(text):
        try:
            rank = int(m.group("rank"))
            title = m.group("title").strip()
            score = int(m.group("score"))
        except Exception:
            continue
        if 1 <= rank <= 50000 and 0 < score <= 100 and title:
            got.setdefault(rank, MetaEntry(rank=rank, title=title, score=score))
    return [got[k] for k in sorted(got.keys())]


def cache_load(path: Path, max_age_days: int) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        ts = float(obj.get("ts", 0))
        age = (time.time() - ts) / 86400.0
        if age > max_age_days:
            return None
        return obj
    except Exception:
        return None


def cache_save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj["ts"] = time.time()
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def recommend_top_n(game_count: int, base_top: int, max_top: int) -> int:
    if game_count <= 3000:
        n = base_top
    elif game_count <= 8000:
        n = max(base_top, 500)
    elif game_count <= 15000:
        n = max(base_top, 750)
    elif game_count <= 25000:
        n = max(base_top, 1000)
    else:
        n = max(base_top, 1500)
    return min(n, max_top)


def scrape_metacritic(system_key: str, url: str, want: int, cache_path: Path,
                      refresh: bool, cache_days: int, delay: float, retries: int, timeout: int) -> List[MetaEntry]:
    if not refresh:
        cached = cache_load(cache_path, cache_days)
        if cached and "entries" in cached:
            return [MetaEntry(**e) for e in cached["entries"]][:want]

    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})

    got: Dict[int, MetaEntry] = {}
    page = 0
    empty_pages = 0
    while len(got) < want and empty_pages < 2:
        page_url = build_page_url(url, page)
        print(f"  - Metacritic page {page}: {page_url}")
        html = fetch_text(sess, page_url, retries, delay, timeout)
        entries = parse_metacritic(html)

        if not entries:
            empty_pages += 1
        else:
            empty_pages = 0

        for e in entries:
            got.setdefault(e.rank, e)

        print(f"    total={len(got)}")
        page += 1
        polite_sleep(delay)
        if page > 500:
            break

    final = [got[k] for k in sorted(got.keys())][:want]
    cache_save(cache_path, {"system": system_key, "entries": [e.__dict__ for e in final]})
    return final


# -----------------------------
# Scan library (recursive), exclude output folder + cache dirs
# -----------------------------

def scan_candidates(system_dir: Path, output_dir: Path) -> List[Candidate]:
    candidates: List[Candidate] = []
    primary_all = {e.lower() for exts in SYSTEM_PRIMARY_EXTS.values() for e in exts}

    for root, dirs, files in os.walk(system_dir):
        rp = Path(root)

        # skip output folder subtree
        try:
            if rp == output_dir or output_dir in rp.parents:
                continue
        except Exception:
            pass

        # skip cache-ish folders
        if rp.name.lower().startswith(".top_games_cache"):
            continue

        for fn in files:
            p = rp / fn
            if p.name.lower().endswith(".nkit.iso"):
                label = p.name[:-9]
                candidates.append(Candidate(p, base_key_for_name(label), label, False))
                continue

            ext = p.suffix.lower()
            is_archive = ext in ARCHIVE_EXTS
            if is_archive or ext in primary_all:
                candidates.append(Candidate(p, base_key_for_name(p.stem), p.stem, is_archive))

    return candidates


def build_index(candidates: List[Candidate]) -> Tuple[List[str], Dict[str, List[int]], Dict[str, List[int]]]:
    labels_norm: List[str] = []
    token_map: Dict[str, List[int]] = {}
    group_map: Dict[str, List[int]] = {}

    for i, c in enumerate(candidates):
        n = normalize_title(c.label)
        labels_norm.append(n)
        for t in [x for x in n.split() if len(x) >= 3]:
            token_map.setdefault(t, []).append(i)
        group_map.setdefault(c.base_key, []).append(i)

    return labels_norm, token_map, group_map


def pick_candidate_indices(title_norm: str, token_map: Dict[str, List[int]], cap: int = 5000) -> List[int]:
    toks = [t for t in title_norm.split() if len(t) >= 4]
    if not toks:
        return []
    toks = sorted(toks, key=lambda t: len(token_map.get(t, [])) or 10**9)[:3]
    sets = [set(token_map.get(t, [])) for t in toks if token_map.get(t)]
    if not sets:
        return []
    inter = set.intersection(*sets) if len(sets) >= 2 else sets[0]
    cand = inter if 1 <= len(inter) <= cap else set.union(*sets)
    if len(cand) > cap:
        cand = set(list(cand)[:cap])
    return list(cand)


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0
    if a in b or b in a:
        return 95.0
    if HAVE_RAPIDFUZZ:
        return float(fuzz.token_set_ratio(a, b))
    aset, bset = set(a.split()), set(b.split())
    if not aset or not bset:
        return 0.0
    return 100.0 * (len(aset & bset) / len(aset | bset))


def best_match(title: str,
               candidates: List[Candidate],
               labels_norm: List[str],
               token_map: Dict[str, List[int]],
               used_indices: set,
               threshold: float) -> Optional[Tuple[int, float]]:
    t = normalize_title(title)

    for i, n in enumerate(labels_norm):
        if i not in used_indices and n == t:
            return i, 100.0

    cand_idxs = pick_candidate_indices(t, token_map)
    if not cand_idxs:
        cand_idxs = list(range(min(4000, len(candidates))))

    best_i = None
    best_s = -1.0
    for i in cand_idxs:
        if i in used_indices:
            continue
        s = similarity(t, labels_norm[i])
        if s > best_s:
            best_s = s
            best_i = i

    if best_i is None or best_s < threshold:
        return None
    return best_i, best_s


# -----------------------------
# Output folder + flat copy helpers
# -----------------------------

def build_output_folder(system_dir: Path) -> Path:
    sys_clean = re.sub(r"[^a-z0-9]+", "", system_dir.name.lower())
    return system_dir / f"top{sys_clean}games"


def unique_dest_path(out_dir: Path, filename: str) -> Path:
    base = Path(filename).stem
    ext = "".join(Path(filename).suffixes)
    cand = out_dir / filename
    if not cand.exists():
        return cand
    for k in range(2, 500):
        cand = out_dir / f"{base} ({k}){ext}"
        if not cand.exists():
            return cand
    return out_dir / f"{base} ({int(time.time())}){ext}"


def copy_flat(src: Path, out_dir: Path, overwrite_files: bool, dry_run: bool) -> str:
    dest = out_dir / src.name
    if dest.exists() and not overwrite_files:
        dest = unique_dest_path(out_dir, src.name)

    if dry_run:
        return f"DRY_COPY -> {dest.name}"

    out_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists() and overwrite_files:
        dest.unlink()
    shutil.copy2(src, dest)
    return f"COPY -> {dest.name}"


# -----------------------------
# 7-Zip support: LIST then extract only selected members
# -----------------------------

def find_7z(sevenzip_arg: Optional[str]) -> Optional[str]:
    if sevenzip_arg and Path(sevenzip_arg).exists():
        return sevenzip_arg
    for exe in ("7z", "7z.exe", "7za", "7za.exe"):
        found = shutil.which(exe)
        if found:
            return found
    for p in (r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if Path(p).exists():
            return p
    return None


@dataclass
class ArchiveMember:
    path: str
    size: int


def sevenz_list_members(sevenz: str, archive: Path, timeout_s: int) -> List[ArchiveMember]:
    # -slt gives stable key/value output
    cmd = [sevenz, "l", "-slt", str(archive)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                       timeout=None if timeout_s <= 0 else timeout_s)
    if r.returncode != 0:
        raise RuntimeError(f"7z list failed: {archive}\n{r.stderr[:800]}")

    members: List[ArchiveMember] = []
    cur_path = None
    cur_size = 0
    cur_is_dir = False

    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("Path = "):
            cur_path = line.replace("Path = ", "", 1).strip()
            cur_size = 0
            cur_is_dir = False
        elif line.startswith("Size = "):
            try:
                cur_size = int(line.replace("Size = ", "", 1).strip())
            except Exception:
                cur_size = 0
        elif line.startswith("Attributes = "):
            # directories often have 'D' attribute
            attrs = line.replace("Attributes = ", "", 1).strip()
            cur_is_dir = ("D" in attrs)
        elif line == "" and cur_path:
            if not cur_is_dir:
                members.append(ArchiveMember(cur_path, cur_size))
            cur_path = None

    if cur_path and not cur_is_dir:
        members.append(ArchiveMember(cur_path, cur_size))

    return members


def select_members_for_system(members: List[ArchiveMember], system_key: str) -> List[str]:
    primary = [e.lower() for e in SYSTEM_PRIMARY_EXTS.get(system_key, [])]
    if not primary:
        # fallback: choose largest non-junk file
        m2 = [m for m in members if Path(m.path).suffix.lower() not in JUNK_EXTS]
        if not m2:
            return []
        return [max(m2, key=lambda x: x.size).path]

    # 1) try best extension in priority order
    for ext in primary:
        hits = [m.path for m in members if m.path.lower().endswith(ext)]
        if hits:
            return hits

    # 2) special: .nkit.iso
    hits = [m.path for m in members if m.path.lower().endswith(".nkit.iso")]
    if hits:
        return hits

    # 3) fallback: biggest non-junk
    m2 = [m for m in members if Path(m.path).suffix.lower() not in JUNK_EXTS]
    if not m2:
        return []
    return [max(m2, key=lambda x: x.size).path]


def sevenz_extract_selected(sevenz: str, archive: Path, selected_members: List[str], dest: Path,
                           timeout_s: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    # extract only selected members
    cmd = [sevenz, "x", "-y", f"-o{str(dest)}", str(archive)] + selected_members
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                   timeout=None if timeout_s <= 0 else timeout_s, check=True)


def extract_zip_selected(archive: Path, selected_members: List[str], dest: Path) -> None:
    import zipfile
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as z:
        for m in selected_members:
            z.extract(m, dest)


def collect_extracted_files(extract_root: Path, selected_members: List[str]) -> List[Path]:
    # selected_members may include subpaths; find them in extract_root
    out: List[Path] = []
    for m in selected_members:
        p = (extract_root / m).resolve()
        if p.exists() and p.is_file():
            out.append(p)

    # Also: if a selected file was a .cue, include referenced bins if present
    cues = [p for p in out if p.suffix.lower() == ".cue"]
    for cue in cues:
        try:
            txt = cue.read_text(errors="ignore")
            refs = re.findall(r'FILE\s+"([^"]+)"', txt, flags=re.IGNORECASE)
            for r in refs:
                rp = (cue.parent / r).resolve()
                if rp.exists() and rp.is_file():
                    out.append(rp)
        except Exception:
            pass

    # dedupe
    seen = set()
    uniq = []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def process_candidate_flat(candidate: Candidate,
                           system_key: str,
                           out_dir: Path,
                           sevenz_path: Optional[str],
                           overwrite_files: bool,
                           dry_run: bool,
                           list_timeout_s: int,
                           extract_timeout_s: int) -> List[str]:
    """
    IMPORTANT FIXES:
      - dry_run never extracts archives
      - archives are listed and only relevant members extracted (not full archive)
    """
    p = candidate.path
    actions: List[str] = []

    if candidate.is_archive:
        ext = p.suffix.lower()

        if ext in (".7z", ".rar") and not sevenz_path:
            actions.append("ERROR: 7z.exe required for .7z/.rar extraction")
            return actions

        # list members
        if ext in (".7z", ".rar"):
            members = sevenz_list_members(sevenz_path, p, timeout_s=list_timeout_s)  # type: ignore[arg-type]
        else:
            # zip: list via python
            import zipfile
            with zipfile.ZipFile(p, "r") as z:
                members = [ArchiveMember(n, 0) for n in z.namelist() if not n.endswith("/")]

        selected = select_members_for_system(members, system_key)
        if not selected:
            actions.append("ARCHIVE: no suitable members found")
            return actions

        if dry_run:
            actions.append(f"DRY_EXTRACT_SELECTED: {', '.join(selected[:5])}{' ...' if len(selected)>5 else ''}")
            return actions

        # extract selected to temp, then flat-copy into out_dir
        with tempfile.TemporaryDirectory(prefix="topgames_extract_") as td:
            extract_root = Path(td)

            try:
                if ext == ".zip":
                    extract_zip_selected(p, selected, extract_root)
                else:
                    sevenz_extract_selected(sevenz_path, p, selected, extract_root, timeout_s=extract_timeout_s)  # type: ignore[arg-type]
            except subprocess.TimeoutExpired:
                return [f"ERROR: extract timeout after {extract_timeout_s}s"]
            except subprocess.CalledProcessError as e:
                return [f"ERROR: 7z extract failed: {str(e)[:200]}"]

            extracted_files = collect_extracted_files(extract_root, selected)
            if not extracted_files:
                return ["EXTRACT_OK but no extracted files resolved"]

            for f in extracted_files:
                actions.append(copy_flat(f, out_dir, overwrite_files, dry_run=False))

        return actions

    # Non-archive: just copy flat; if cue, also copy referenced bins
    actions.append(copy_flat(p, out_dir, overwrite_files, dry_run))
    if p.suffix.lower() == ".cue":
        try:
            txt = p.read_text(errors="ignore")
            refs = re.findall(r'FILE\s+"([^"]+)"', txt, flags=re.IGNORECASE)
            for r in refs:
                rp = (p.parent / r).resolve()
                if rp.exists() and rp.is_file():
                    actions.append(copy_flat(rp, out_dir, overwrite_files, dry_run))
        except Exception:
            pass
    return actions


def write_report(report_path: Path, rows: List[dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        report_path.write_text("no rows\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# -----------------------------
# Main per-system processing (Metacritic-only in this v2 to keep your loop stable)
# -----------------------------

def process_system(system_dir: Path,
                   system_key: str,
                   cache_root: Path,
                   sevenz_path: Optional[str],
                   base_top: int,
                   max_top: int,
                   mc_threshold: float,
                   overwrite_output_folder: bool,
                   overwrite_files_in_output: bool,
                   refresh: bool,
                   cache_days: int,
                   delay: float,
                   retries: int,
                   timeout: int,
                   dry_run: bool,
                   list_timeout_s: int,
                   extract_timeout_s: int) -> None:

    out_dir = build_output_folder(system_dir)

    if out_dir.exists() and overwrite_output_folder:
        if dry_run:
            print(f"[{system_dir.name}] DRY: would delete output folder: {out_dir}")
        else:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{system_dir.name}] Output: {out_dir}")

    candidates = scan_candidates(system_dir, out_dir)
    if not candidates:
        print(f"[{system_dir.name}] No candidate ROM/archive files found.\n")
        return

    labels_norm, token_map, group_map = build_index(candidates)
    game_count = len(candidates)

    titles: List[Tuple[str, int, int]] = []

    mc_url = METACRITIC_URL.get(system_key)
    if mc_url:
        want = recommend_top_n(game_count, base_top, max_top)
        print(f"[{system_dir.name}] Metacritic target: {want} (candidates={game_count})")
        mc_cache = cache_root / "metacritic" / f"{system_key}.json"
        try:
            entries = scrape_metacritic(system_key, mc_url, want, mc_cache, refresh, cache_days, delay, retries, timeout)
            titles = [(e.title, e.rank, e.score) for e in entries]
        except Exception as e:
            print(f"[{system_dir.name}] Metacritic scrape failed: {e}")
            return
    else:
        print(f"[{system_dir.name}] No Metacritic mapping; skipping.\n")
        return

    used_indices = set()
    report_rows: List[dict] = []

    print(f"[{system_dir.name}] Matching & exporting (flat)...")
    for title, rank, score in titles:
        match = best_match(title, candidates, labels_norm, token_map, used_indices, mc_threshold)
        if not match:
            report_rows.append({
                "rank": rank, "metascore": score, "title": title,
                "status": "missing", "matched_paths": "", "match_score": "", "actions": ""
            })
            continue

        idx, ms = match
        used_indices.add(idx)

        base_key = candidates[idx].base_key
        companion_idxs = group_map.get(base_key, [])
        to_process = [idx]
        for ci in companion_idxs:
            if ci not in used_indices:
                used_indices.add(ci)
                to_process.append(ci)

        all_actions = []
        matched_paths = []
        for i2 in to_process:
            c = candidates[i2]
            matched_paths.append(str(c.path))
            try:
                acts = process_candidate_flat(
                    c, system_key, out_dir,
                    sevenz_path=sevenz_path,
                    overwrite_files=overwrite_files_in_output,
                    dry_run=dry_run,
                    list_timeout_s=list_timeout_s,
                    extract_timeout_s=extract_timeout_s,
                )
                all_actions.extend(acts)
            except Exception as e:
                all_actions.append(f"ERROR: {e}")

        report_rows.append({
            "rank": rank, "metascore": score, "title": title,
            "status": "matched",
            "matched_paths": " | ".join(matched_paths[:4]) + (" ..." if len(matched_paths) > 4 else ""),
            "match_score": f"{ms:.1f}",
            "actions": " ; ".join(all_actions[:10]) + (" ..." if len(all_actions) > 10 else "")
        })

    write_report(out_dir / "_report.csv", report_rows)
    print(f"[{system_dir.name}] Done. Report: {out_dir / '_report.csv'}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library-root", required=True)
    ap.add_argument("--sevenzip", default=None)

    ap.add_argument("--base-top", type=int, default=300)
    ap.add_argument("--max-top", type=int, default=2000)
    ap.add_argument("--metacritic-threshold", type=float, default=72.0)

    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cache-days", type=int, default=14)

    ap.add_argument("--delay", type=float, default=1.25)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=25)

    ap.add_argument("--no-overwrite", action="store_true")
    ap.add_argument("--overwrite-files", action="store_true")

    ap.add_argument("--dry-run", action="store_true")

    # NEW: prevent hangs
    ap.add_argument("--list-timeout", type=int, default=60, help="Timeout for 7z list (seconds)")
    ap.add_argument("--extract-timeout", type=int, default=0, help="Timeout for 7z extract (seconds). 0 = no timeout")

    # speed
    ap.add_argument("--no-reddit", action="store_true", help="(kept for compatibility; reddit removed from v2)")
    args = ap.parse_args()

    root = Path(args.library_root).expanduser().resolve()
    if not root.exists():
        print(f"Library root not found: {root}", file=sys.stderr)
        return 2

    sevenz = find_7z(args.sevenzip)
    if not sevenz:
        print("NOTE: 7z.exe not found. .7z/.rar will be skipped unless you install 7-Zip or pass --sevenzip.", file=sys.stderr)

    cache_root = root / ".top_games_cache"

    # Process ALL first-level dirs, but skip obvious non-systems
    system_dirs = [p for p in root.iterdir() if p.is_dir()]

    def should_skip_dir(p: Path) -> bool:
        n = p.name.strip()
        low = n.lower()
        if low.startswith("."):
            return True
        if low in {"top games", ".top_games_cache"}:
            return True
        if low.startswith("top") and low.endswith("games"):
            return True
        if canon_system(n) == "BIOS":
            return True
        return False

    system_dirs = [p for p in system_dirs if not should_skip_dir(p)]

    print(f"Library root: {root}")
    print(f"Systems found: {', '.join([p.name for p in system_dirs])}")
    print(f"Overwrite output folders: {'NO' if args.no_overwrite else 'YES'}")
    print("")

    for sd in sorted(system_dirs, key=lambda x: x.name.lower()):
        key = canon_system(sd.name)
        process_system(
            system_dir=sd,
            system_key=key,
            cache_root=cache_root,
            sevenz_path=sevenz,
            base_top=args.base_top,
            max_top=args.max_top,
            mc_threshold=args.metacritic_threshold,
            overwrite_output_folder=(not args.no_overwrite),
            overwrite_files_in_output=args.overwrite_files,
            refresh=args.refresh,
            cache_days=args.cache_days,
            delay=args.delay,
            retries=args.retries,
            timeout=args.timeout,
            dry_run=args.dry_run,
            list_timeout_s=args.list_timeout,
            extract_timeout_s=args.extract_timeout,
        )

    print("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
