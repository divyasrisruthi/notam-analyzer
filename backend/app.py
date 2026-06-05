"""
NOTAM Waypoint Analyzer - Backend
Secure Flask API with KD-tree nearest-waypoint lookup
"""


import os
import re
import csv
import logging
import secrets
import time
from pathlib import Path
from functools import wraps
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt, degrees, atan2, pi


from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from scipy.spatial import cKDTree
import numpy as np


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── App Setup ─────────────────────────────────────────────────────────────────
STATIC_FOLDER = str(Path(__file__).parent.parent / "frontend")
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path="")
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "http://localhost:5000").split(","))
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


# ── Rate Limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", 30))
_rate_store: dict[str, list] = defaultdict(list)


def rate_limit(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        ip = request.headers.get("X-Forwarded-For", request.remote_addr).split(",")[0].strip()
        now = time.time()
        _rate_store[ip] = [t for t in _rate_store[ip] if now - t < 60]
        if len(_rate_store[ip]) >= RATE_LIMIT:
            log.warning("Rate limit hit for IP %s", ip)
            return jsonify({"error": "Too many requests. Please wait."}), 429
        _rate_store[ip].append(now)
        return f(*args, **kwargs)
    return wrapped


# ── Waypoint Database ─────────────────────────────────────────────────────────
WPT_NAMES:    list[str]   = []
WPT_COORDS:   list[tuple] = []
WPT_META:     list[dict]  = []
WPT_AIRWAYS:  dict        = defaultdict(list)   # name  → [idx, ...]
WPT_BY_AIRWAY: dict       = defaultdict(list)   # airway → [idx, ...] in CSV order
_kdtree: cKDTree | None   = None


def _to_xyz(lat_deg: float, lon_deg: float):
    lat, lon = radians(lat_deg), radians(lon_deg)
    return cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)


def load_waypoints(csv_path: str) -> int:
    global WPT_NAMES, WPT_COORDS, WPT_META, WPT_AIRWAYS, WPT_BY_AIRWAY, _kdtree

    path = Path(csv_path)
    if not path.exists():
        log.error("Waypoint file not found: %s", csv_path)
        return len(WPT_NAMES)  # ✅ KEEP existing data

    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    except Exception:
        log.exception("Failed to read waypoint CSV: %s", csv_path)
        return len(WPT_NAMES)  # ✅ KEEP existing data

    if not rows:
        log.error("CSV is empty: %s", csv_path)
        return len(WPT_NAMES)  # ✅ KEEP existing data

    header = [c.strip().upper() for c in rows[0]]
    log.info("CSV header: %s", header)

    names, coords, meta = [], [], []
    skipped = 0

    # ---------- FORMAT DETECTION ----------
    is_your_format = (
        len(header) >= 4
        and header[0] in ("AWID", "AWY", "AIRWAY", "AWY_ID", "AIRWAY_ID")
        and header[1] in ("NDID", "WPT", "WAYPOINT", "FIX", "NAME", "IDENT")
        and header[2] in ("NDIC", "FIR", "COUNTRY", "REGION", "ICAO")
    )

    # ✅ Backup detection from sample rows
    if not is_your_format and len(rows) >= 5:
        for test_row in rows[1:10]:
            if len(test_row) >= 4:
                sample = test_row[3].strip().upper().replace(" ", "")
                if (
                    re.match(r'^\d{6}[NS]\d{7}[EW]$', sample) or
                    re.match(r'^[NS]\d{6}[EW]\d{7}$', sample) or
                    re.match(r'^[NS]\d{4}[EW]\d{5}$', sample) or
                    re.match(r'^\d{4}[NS]\d{5}[EW]$', sample)
                ):
                    is_your_format = True
                    break

    log.info("Detected format: %s", "AIRWAY/FIX/FIR/COORD" if is_your_format else "GENERIC LAT/LON")

    # ---------- FORMAT 1 ----------
    if is_your_format:
        for row in rows[1:]:
            try:
                if len(row) < 4:
                    skipped += 1
                    continue

                airway_id = row[0].strip().upper()
                wpt_name = row[1].strip().upper()
                fir_code = row[2].strip().upper()
                coord_raw = row[3].strip().upper().replace(" ", "")

                # skip separators / junk
                if not wpt_name or not coord_raw:
                    skipped += 1
                    continue

                if re.match(r'^[-]+$', wpt_name) or re.match(r'^[-]+$', airway_id):
                    skipped += 1
                    continue

                parsed = _parse_coord_str(coord_raw)
                if parsed is None:
                    skipped += 1
                    continue

                lat, lon = parsed

                names.append(wpt_name)
                coords.append((lat, lon))
                meta.append({
                    "name": wpt_name,
                    "airway": airway_id,
                    "fir": fir_code,
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "coord_raw": coord_raw
                })

            except Exception:
                skipped += 1
                continue

        log.info("Skipped %d bad rows", skipped)

    # ---------- FORMAT 2 ----------
    else:
        col_map = {c.lower(): i for i, c in enumerate(rows[0])}

        name_col = next((col_map[h] for h in ("name","ndid","ident","id","fix","waypoint","wpt") if h in col_map), None)
        lat_col  = next((col_map[h] for h in ("latitude","lat","lat_deg") if h in col_map), None)
        lon_col  = next((col_map[h] for h in ("longitude","lon","long","lng","lon_deg") if h in col_map), None)

        if not all(c is not None for c in [name_col, lat_col, lon_col]):
            log.error("Cannot detect LAT/LON columns. Headers found: %s", header)
            return len(WPT_NAMES)  # ✅ KEEP existing data

        for row in rows[1:]:
            try:
                name = row[name_col].strip().upper()
                lat = float(row[lat_col])
                lon = float(row[lon_col])

                if not name or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    skipped += 1
                    continue

                names.append(name)
                coords.append((lat, lon))
                meta.append({
                    "name": name,
                    "airway": "",
                    "fir": "",
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "coord_raw": ""
                })

            except Exception:
                skipped += 1
                continue

        log.info("Skipped %d bad rows", skipped)

    # ---------- FINAL VALIDATION ----------
    if not names or len(names) < 50:
        log.error(
            "Rejected dataset: only %d valid rows from %s. Keeping previous dataset.",
            len(names), csv_path
        )
        return len(WPT_NAMES)  # ✅ DO NOT overwrite working data

    # ✅ Build KD-tree BEFORE replacing
    try:
        xyz = np.array([_to_xyz(lat, lon) for lat, lon in coords])
        new_tree = cKDTree(xyz)
    except Exception:
        log.exception("KD-tree build failed. Keeping previous dataset.")
        return len(WPT_NAMES)

    # ✅ ONLY NOW overwrite
    WPT_NAMES[:] = names
    WPT_COORDS[:] = coords
    WPT_META[:] = meta

    WPT_AIRWAYS.clear()
    WPT_BY_AIRWAY.clear()

    for i, name in enumerate(names):
        WPT_AIRWAYS[name].append(i)
        airway = meta[i].get("airway", "")
        if airway:
            WPT_BY_AIRWAY[airway].append(i)

    _kdtree = new_tree

    log.info(
        "✅ Loaded %d waypoints | unique=%d | airways=%d",
        len(names), len(WPT_AIRWAYS), len(WPT_BY_AIRWAY)
    )

    return len(names)


# ── Coordinate parsers ────────────────────────────────────────────────────────
def _parse_coord_str(s: str) -> tuple[float, float] | None:
    s = s.strip().upper().replace(" ", "")
    patterns = [
        (r'^(\d{2})(\d{2})(\d{2})([NS])(\d{3})(\d{2})(\d{2})([EW])$', True),
        (r'^([NS])(\d{2})(\d{2})(\d{2})([EW])(\d{3})(\d{2})(\d{2})$', False),
        (r'^([NS])(\d{2})(\d{2})(\d{2})([EW])(\d{2})(\d{2})(\d{2})$', False),
        (r'^(\d{2})(\d{2})([NS])(\d{3})(\d{2})([EW])$',               True),
        (r'^([NS])(\d{2})(\d{2})([EW])(\d{3})(\d{2})$',               False),
        (r'^(\d{2})(\d{2})([NS])(\d{3})(\d{2})([EW])$',               True),
    ]
    for pat, lat_first in patterns:
        m = re.match(pat, s)
        if not m:
            continue
        g = m.groups()
        if lat_first:
            if len(g) == 8:   # DDMMSSNDDDMMSSe
                lat = int(g[0]) + int(g[1])/60 + int(g[2])/3600
                lon = int(g[4]) + int(g[5])/60 + int(g[6])/3600
                if g[3]=='S': lat=-lat
                if g[7]=='W': lon=-lon
            else:              # DDMMNDDDMMe
                lat = int(g[0]) + int(g[1])/60
                lon = int(g[3]) + int(g[4])/60
                if g[2]=='S': lat=-lat
                if g[5]=='W': lon=-lon
        else:
            if len(g) == 8:   # NDDMMSSEDDDMMSSe
                lat = int(g[1]) + int(g[2])/60 + int(g[3])/3600
                lon = int(g[5]) + int(g[6])/60 + int(g[7])/3600
                if g[0]=='S': lat=-lat
                if g[4]=='W': lon=-lon
            else:              # NDDMMEDDDmm
                lat = int(g[1]) + int(g[2])/60
                lon = int(g[4]) + int(g[5])/60
                if g[0]=='S': lat=-lat
                if g[3]=='W': lon=-lon
        return round(lat, 6), round(lon, 6)
    return None


def parse_coord(coord_str: str) -> dict | None:
    result = _parse_coord_str(coord_str)
    return {"lat": result[0], "lon": result[1]} if result else None


# ── Geo math ──────────────────────────────────────────────────────────────────
def _bearing(lat1, lon1, lat2, lon2) -> float:
    φ1, φ2 = radians(lat1), radians(lat2)
    Δλ = radians(lon2 - lon1)
    y = sin(Δλ) * cos(φ2)
    x = cos(φ1) * sin(φ2) - sin(φ1) * cos(φ2) * cos(Δλ)
    return (degrees(atan2(y, x)) + 360) % 360


def _dist_nm(lat1, lon1, lat2, lon2) -> float:
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((φ2-φ1)/2)**2 + cos(φ1)*cos(φ2)*sin((λ2-λ1)/2)**2
    return (180/pi) * 60 * 2 * asin(sqrt(a))


def _destination_point(lat, lon, bearing_deg, distance_nm) -> tuple[float, float]:
    d   = distance_nm / (180 * 60)
    lat1 = radians(lat); lon1 = radians(lon); brng = radians(bearing_deg)
    lat2 = asin(sin(lat1)*cos(d) + cos(lat1)*sin(d)*cos(brng))
    lon2 = lon1 + atan2(sin(brng)*sin(d)*cos(lat1), cos(d)-sin(lat1)*sin(lat2))
    return round(degrees(lat2), 6), round(degrees(lon2), 6)


def nearest_waypoints(lat, lon, k=5) -> list[dict]:
    if _kdtree is None:
        return []
    k = min(k, len(WPT_NAMES))
    xyz_q = np.array(_to_xyz(lat, lon))
    dists, idxs = _kdtree.query(xyz_q, k=k)
    if k == 1:
        dists, idxs = [dists], [idxs]
    results = []
    for d3, idx in zip(dists, idxs):
        chord = min(d3, 2.0)
        nm = 2 * asin(chord/2) * 180/pi * 60
        results.append({**WPT_META[idx], "distance_nm": round(nm, 2)})
    return results


def get_fir_by_distance(lat: float, lon: float, threshold_nm: float = 50.0) -> str:
    """
    Determine the FIR code (country) based on nearby waypoints.
    Uses distance-based majority voting among nearby waypoints.
   
    This handles cases where waypoints are near borders of countries/FIRs.
    For example, OBDEG (ZL) is near ADMUX (ZL) and others, so it may be
    correctly classified, but if surrounded by a different FIR, that one
    takes precedence based on what's most common nearby.
   
    Args:
        lat: Latitude of the waypoint
        lon: Longitude of the waypoint
        threshold_nm: Distance threshold for considering nearby waypoints (default 50 NM)
   
    Returns:
        FIR code if found via distance voting, otherwise empty string
    """
    if _kdtree is None:
        return ""
   
    # Get nearby waypoints (up to 10)
    nearby = nearest_waypoints(lat, lon, k=10)
    if not nearby:
        return ""
   
    # Find all FIR codes within the distance threshold
    fir_codes = {}
    for wp in nearby:
        dist = wp.get("distance_nm", 999)
        if dist > threshold_nm:
            continue
        fir = wp.get("fir", "").strip()
        if not fir:
            continue
        fir_codes[fir] = fir_codes.get(fir, 0) + 1
   
    if fir_codes:
        # Use the most common FIR code
        best_fir = max(fir_codes.items(), key=lambda x: x[1])[0]
        if best_fir:
            log.debug("get_fir_by_distance: Using FIR '%s' from %d nearby waypoints within %.1f NM",
                     best_fir, len(nearby), threshold_nm)
            return best_fir
   
    return ""


def _resolve_dist_dir_on_airway(
    airway,
    anchor_name,
    anchor_lat,
    anchor_lon,
    target_dist_nm,
    bearing_deg,
):
    """
    Resolve a distance/direction point by WALKING the airway in sequence,
    not by globally sorting all waypoints by radial distance.

    Example:
        LIKMI - 60KM WEST OF LIKMI
    If BATUS is the first waypoint west of LIKMI and lies beyond 32 NM,
    return BATUS (not a farther point like RIMDU).
    """

    if airway not in WPT_BY_AIRWAY:
        return None

    airway_indices = WPT_BY_AIRWAY[airway]
    anchor_name_u = anchor_name.strip().upper()

    # Find ALL occurrences of anchor on this airway
    anchor_positions = [
        pos for pos, idx in enumerate(airway_indices)
        if WPT_META[idx]["name"].strip().upper() == anchor_name_u
    ]

    if not anchor_positions:
        return None

    def ang_diff(a, b):
        return abs(((a - b) + 180) % 360 - 180)

    best_choice = None
    best_score = None

    # Decide best anchor occurrence + walk direction
    for anchor_pos in anchor_positions:
        for step in (-1, 1):
            next_pos = anchor_pos + step
            if not (0 <= next_pos < len(airway_indices)):
                continue

            next_idx = airway_indices[next_pos]
            next_meta = WPT_META[next_idx]
            next_lat, next_lon = WPT_COORDS[next_idx]

            brg = _bearing(anchor_lat, anchor_lon, next_lat, next_lon)
            score = ang_diff(brg, bearing_deg)

            if best_score is None or score < best_score:
                best_score = score
                best_choice = (anchor_pos, step)

    if best_choice is None:
        return None

    anchor_pos, step = best_choice

    # Walk cumulatively along airway
    total_dist = 0.0
    prev_lat, prev_lon = anchor_lat, anchor_lon
    prev_name = anchor_name_u

    pos = anchor_pos + step
    last_valid = None

    while 0 <= pos < len(airway_indices):
        idx = airway_indices[pos]
        meta = WPT_META[idx]
        wlat, wlon = WPT_COORDS[idx]

        leg_dist = _dist_nm(prev_lat, prev_lon, wlat, wlon)
        total_dist += leg_dist

        current = {
            **meta,
            "distance_nm": round(total_dist, 2),
            "leg_distance_nm": round(leg_dist, 2),
            "from_name": prev_name,
        }

        last_valid = current

        print(
    f"[AIRWAY WALK] airway={airway} anchor={anchor_name_u} "
    f"step={step} visiting={meta['name']} leg={leg_dist:.1f}NM total={total_dist:.1f}NM target={target_dist_nm:.1f}NM"
)
        # FIRST waypoint whose cumulative airway distance crosses target
        if total_dist >= target_dist_nm:
            return current

        prev_lat, prev_lon = wlat, wlon
        prev_name = meta["name"].strip().upper()
        pos += step

    # fallback = furthest reachable on this branch
    return last_valid

def _pick_waypoint_index(name: str, route: str = "", fir_hint: str = ""):
    """
    Pick the correct waypoint occurrence when the same name exists multiple times
    (example: HAM in ZW / OL / ED).

    Priority:
      1) same route/airway
      2) same FIR prefix
      3) fallback to first match
    """
    name_u = name.strip().upper()
    idxs = WPT_AIRWAYS.get(name_u, [])

    if not idxs:
        return None

    route_u = (route or "").strip().upper()
    fir_u = (fir_hint or "").strip().upper()

    # 1) Prefer same airway / route
    if route_u:
        route_matches = [
            idx for idx in idxs
            if WPT_META[idx].get("airway", "").strip().upper() == route_u
        ]
        if route_matches:
            # if FIR hint exists, refine further
            if fir_u:
                fir_matches = [
                    idx for idx in route_matches
                    if WPT_META[idx].get("fir", "").strip().upper().startswith(fir_u[:2])
                ]
                if fir_matches:
                    return fir_matches[0]
            return route_matches[0]

    # 2) Otherwise prefer FIR match
    if fir_u:
        fir_matches = [
            idx for idx in idxs
            if WPT_META[idx].get("fir", "").strip().upper().startswith(fir_u[:2])
        ]
        if fir_matches:
            return fir_matches[0]

    # 3) Fallback
    return idxs[0]

def _nearest_waypoint_on_route(route: str, lat: float, lon: float, exclude_name: str = ""):
    """
    Find the nearest waypoint on the same airway/route to a computed point.
    Used for dist_dir -> dist_dir cases.
    """
    route_u = (route or "").strip().upper()
    exclude_u = (exclude_name or "").strip().upper()

    idxs = WPT_BY_AIRWAY.get(route_u, [])
    if not idxs:
        return None

    best = None
    best_dist = None

    for idx in idxs:
        meta = WPT_META[idx]
        name_u = meta["name"].strip().upper()

        if exclude_u and name_u == exclude_u:
            continue

        wlat, wlon = WPT_COORDS[idx]
        d = _dist_nm(lat, lon, wlat, wlon)

        if best_dist is None or d < best_dist:
            best_dist = d
            best = {
                **meta,
                "distance_nm": round(d, 2)
            }

    return best
def _get_airway_blocks_in_fir(route: str, fir_hint: str):
    """
    Return contiguous waypoint blocks for a given airway inside the NOTAM FIR.

    Example:
        W189 has FIR sequence:
        UE UE UE UE UE UE ZW ZW ZW ZW SE

        For fir_hint='ZW', return one block:
        AKLAS -> DAKPA

    Returns:
        list of tuples: [(start_meta, end_meta), ...]
    """
    route_u = (route or "").strip().upper()
    fir_u = (fir_hint or "").strip().upper()[:2]

    idxs = WPT_BY_AIRWAY.get(route_u, [])
    if not idxs or not fir_u:
        return []

    # positions on this airway that belong to the target FIR
    matching_positions = [
        pos for pos, idx in enumerate(idxs)
        if WPT_META[idx].get("fir", "").strip().upper().startswith(fir_u)
    ]

    if not matching_positions:
        return []

    # split into contiguous position blocks
    blocks = []
    block_start = matching_positions[0]
    block_end = matching_positions[0]

    for pos in matching_positions[1:]:
        if pos == block_end + 1:
            block_end = pos
        else:
            start_idx = idxs[block_start]
            end_idx = idxs[block_end]
            blocks.append((WPT_META[start_idx], WPT_META[end_idx]))
            block_start = pos
            block_end = pos

    # add final block
    start_idx = idxs[block_start]
    end_idx = idxs[block_end]
    blocks.append((WPT_META[start_idx], WPT_META[end_idx]))

    return blocks

def _first_waypoint_in_direction_on_airway(route: str, anchor_name: str, bearing_deg: float):
    """
    On a given airway, find the first waypoint encountered from anchor_name
    in the direction that best matches bearing_deg.

    Returns:
        (first_meta, first_lat, first_lon, distance_from_anchor_nm)
    or
        (None, None, None, None)
    """
    route_u = (route or "").strip().upper()
    anchor_u = (anchor_name or "").strip().upper()

    idxs = WPT_BY_AIRWAY.get(route_u, [])
    if not idxs:
        return None, None, None, None

    anchor_positions = [
        pos for pos, idx in enumerate(idxs)
        if WPT_META[idx]["name"].strip().upper() == anchor_u
    ]
    if not anchor_positions:
        return None, None, None, None

    def ang_diff(a, b):
        return abs(((a - b) + 180) % 360 - 180)

    best_choice = None
    best_score = None

    for anchor_pos in anchor_positions:
        for step in (-1, 1):
            next_pos = anchor_pos + step
            if not (0 <= next_pos < len(idxs)):
                continue

            next_idx = idxs[next_pos]
            next_lat, next_lon = WPT_COORDS[next_idx]
            anchor_idx = idxs[anchor_pos]
            anchor_lat, anchor_lon = WPT_COORDS[anchor_idx]

            brg = _bearing(anchor_lat, anchor_lon, next_lat, next_lon)
            score = ang_diff(brg, bearing_deg)

            if best_score is None or score < best_score:
                best_score = score
                best_choice = (anchor_pos, step)

    if best_choice is None:
        return None, None, None, None

    anchor_pos, step = best_choice
    next_pos = anchor_pos + step
    if not (0 <= next_pos < len(idxs)):
        return None, None, None, None

    anchor_idx = idxs[anchor_pos]
    first_idx = idxs[next_pos]

    anchor_lat, anchor_lon = WPT_COORDS[anchor_idx]
    first_lat, first_lon = WPT_COORDS[first_idx]
    first_meta = WPT_META[first_idx]

    dist_nm = _dist_nm(anchor_lat, anchor_lon, first_lat, first_lon)

    return first_meta, first_lat, first_lon, dist_nm

def _locate_dist_dir_leg_on_airway(route: str, anchor_name: str, target_dist_nm: float, bearing_deg: float):
    """
    Locate which airway leg contains the geometric dist_dir point.

    Returns a dict like:
    {
        "anchor_pos": int,
        "step": -1 or 1,
        "from_pos": int,
        "to_pos": int,
        "from_meta": {...},
        "to_meta": {...},
        "distance_before_leg_nm": float,
        "leg_distance_nm": float,
    }

    Meaning the dist_dir point lies somewhere on the leg:
        from_meta <-> to_meta
    """
    route_u = (route or "").strip().upper()
    anchor_u = (anchor_name or "").strip().upper()

    idxs = WPT_BY_AIRWAY.get(route_u, [])
    if not idxs:
        return None

    anchor_positions = [
        pos for pos, idx in enumerate(idxs)
        if WPT_META[idx]["name"].strip().upper() == anchor_u
    ]
    if not anchor_positions:
        return None

    def ang_diff(a, b):
        return abs(((a - b) + 180) % 360 - 180)

    best_choice = None
    best_score = None

    # choose anchor occurrence + walking direction using first adjacent leg bearing
    for anchor_pos in anchor_positions:
        for step in (-1, 1):
            next_pos = anchor_pos + step
            if not (0 <= next_pos < len(idxs)):
                continue

            anchor_idx = idxs[anchor_pos]
            next_idx = idxs[next_pos]

            a_lat, a_lon = WPT_COORDS[anchor_idx]
            n_lat, n_lon = WPT_COORDS[next_idx]

            brg = _bearing(a_lat, a_lon, n_lat, n_lon)
            score = ang_diff(brg, bearing_deg)

            if best_score is None or score < best_score:
                best_score = score
                best_choice = (anchor_pos, step)

    if best_choice is None:
        return None

    anchor_pos, step = best_choice
    anchor_idx = idxs[anchor_pos]
    anchor_lat, anchor_lon = WPT_COORDS[anchor_idx]

    total_dist = 0.0
    prev_pos = anchor_pos
    prev_idx = anchor_idx
    prev_lat, prev_lon = anchor_lat, anchor_lon

    pos = anchor_pos + step

    while 0 <= pos < len(idxs):
        idx = idxs[pos]
        wlat, wlon = WPT_COORDS[idx]
        leg_dist = _dist_nm(prev_lat, prev_lon, wlat, wlon)

        # dist_dir point lies on this leg
        if total_dist <= target_dist_nm <= total_dist + leg_dist:
            return {
                "anchor_pos": anchor_pos,
                "step": step,
                "from_pos": prev_pos,
                "to_pos": pos,
                "from_meta": WPT_META[prev_idx],
                "to_meta": WPT_META[idx],
                "distance_before_leg_nm": round(total_dist, 2),
                "leg_distance_nm": round(leg_dist, 2),
            }

        total_dist += leg_dist
        prev_pos = pos
        prev_idx = idx
        prev_lat, prev_lon = wlat, wlon
        pos += step

    # If the target goes beyond the route end, use the last reachable leg
    if prev_pos != anchor_pos:
        edge_from = min(anchor_pos, prev_pos)
        edge_to = max(anchor_pos, prev_pos)
        return {
            "anchor_pos": anchor_pos,
            "step": step,
            "from_pos": edge_from,
            "to_pos": edge_to,
            "from_meta": WPT_META[idxs[edge_from]],
            "to_meta": WPT_META[idxs[edge_to]],
            "distance_before_leg_nm": round(total_dist, 2),
            "leg_distance_nm": 0.0,
        }

    return None

def _resolve_dist_dir_against_waypoint_on_route(
    route: str,
    anchor_name: str,
    target_dist_nm: float,
    bearing_deg: float,
    fixed_name: str,
):
    """
    Resolve a dist_dir boundary against a fixed waypoint on the same route.

    POLICY:
    - If the computed boundary lies inside a leg, include that whole touched leg.
    - Do not include any extra untouched leg.
    - If the fixed endpoint lies farther away from the anchor in the walking direction,
      snap to the anchor-side endpoint of the containing leg.
    - Otherwise snap to the far-side endpoint of the containing leg.
    - If the boundary lands exactly on a waypoint, use that exact waypoint.
    """

    leg = _locate_dist_dir_leg_on_airway(
        route=route,
        anchor_name=anchor_name,
        target_dist_nm=target_dist_nm,
        bearing_deg=bearing_deg,
    )
    if not leg:
        return None

    fixed_u = (fixed_name or "").strip().upper()
    fixed_positions = _route_positions(route, fixed_u)
    if not fixed_positions:
        return None

    anchor_pos = leg["anchor_pos"]
    step = leg["step"]

    # Choose the occurrence of the fixed waypoint closest in route order to the chosen anchor side
    other_pos = min(fixed_positions, key=lambda p: abs(p - anchor_pos))

    # Exact-hit tolerance: if boundary is effectively on the "to" waypoint, use that waypoint
    EPS_NM = 0.25
    leg_end_dist = leg["distance_before_leg_nm"] + leg["leg_distance_nm"]

    if abs(leg_end_dist - target_dist_nm) <= EPS_NM:
        snapped = {
            **leg["to_meta"],
            "distance_nm": round(leg_end_dist, 2),
            "leg_distance_nm": round(leg["leg_distance_nm"], 2),
            "snap_reason": "exact waypoint match"
        }
        return snapped

    # If fixed endpoint lies farther along the walking direction away from anchor,
    # closure is from boundary -> fixed endpoint, so include full containing leg
    # by snapping to anchor-side endpoint of that leg.
    fixed_ahead_of_anchor = (
        (step == 1 and other_pos > anchor_pos) or
        (step == -1 and other_pos < anchor_pos)
    )

    if fixed_ahead_of_anchor:
        snapped = {
            **leg["from_meta"],
            "distance_nm": round(leg["distance_before_leg_nm"], 2),
            "leg_distance_nm": round(leg["leg_distance_nm"], 2),
            "snap_reason": "enclosing touched leg from anchor side"
        }
    else:
        snapped = {
            **leg["to_meta"],
            "distance_nm": round(leg_end_dist, 2),
            "leg_distance_nm": round(leg["leg_distance_nm"], 2),
            "snap_reason": "enclosing touched leg from far side"
        }

    return snapped

def _route_positions(route: str, name: str):
    """
    Return all positions of a waypoint name on a route in CSV order.
    """
    route_u = (route or "").strip().upper()
    name_u = (name or "").strip().upper()

    idxs = WPT_BY_AIRWAY.get(route_u, [])
    return [
        pos for pos, idx in enumerate(idxs)
        if WPT_META[idx]["name"].strip().upper() == name_u
    ]

def _locate_coord_leg_on_airway(route: str, lat: float, lon: float):
    """
    Find the airway leg on 'route' that best contains / matches a raw coordinate.
    Uses detour error:
        dist(A, coord) + dist(coord, B) - dist(A, B)
    Smallest value means coord lies closest to that leg.
    """
    route_u = (route or "").strip().upper()
    idxs = WPT_BY_AIRWAY.get(route_u, [])
    if len(idxs) < 2:
        return None

    best = None
    best_metric = None

    for pos in range(len(idxs) - 1):
        idx_a = idxs[pos]
        idx_b = idxs[pos + 1]

        a_lat, a_lon = WPT_COORDS[idx_a]
        b_lat, b_lon = WPT_COORDS[idx_b]

        d1 = _dist_nm(a_lat, a_lon, lat, lon)
        d2 = _dist_nm(lat, lon, b_lat, b_lon)
        leg = _dist_nm(a_lat, a_lon, b_lat, b_lon)

        detour = abs((d1 + d2) - leg)

        if best_metric is None or detour < best_metric:
            best_metric = detour
            best = {
                "from_pos": pos,
                "to_pos": pos + 1,
                "from_meta": WPT_META[idx_a],
                "to_meta": WPT_META[idx_b],
                "detour_nm": round(detour, 2),
                "leg_distance_nm": round(leg, 2),
            }

    return best

def _pick_position_toward_other(route: str, candidate_positions: list[int], other_name: str):
    """
    Among candidate waypoint positions, pick the one closest in route-order
    to the other endpoint's route position.
    """
    if not candidate_positions:
        return None

    other_positions = _route_positions(route, other_name)
    if not other_positions:
        return candidate_positions[0]

    best_pos = None
    best_gap = None

    for cp in candidate_positions:
        gap = min(abs(cp - op) for op in other_positions)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_pos = cp

    return best_pos


def resolve_dist_dir_point(route, point, other_point=None, fir_hint=""):
    """
    Resolve one dist_dir endpoint.

    Returns ALWAYS:
        (best, dest_lat, dest_lon, anchor_tuple)
    where:
        best = snapped waypoint dict or None
        dest_lat/dest_lon = computed geometric point or None/None
        anchor_tuple = (fix_lat, fix_lon) or None
    """

    try:
        fix_name = point.get("fix_name", "").strip().upper()
        if not fix_name:
            log.warning("resolve_dist_dir_point: missing fix_name in point=%s", point)
            return None, None, None, None

        fix_idx = _pick_waypoint_index(fix_name, route=route, fir_hint=fir_hint)
        if fix_idx is None:
            log.warning(
                "resolve_dist_dir_point: could not resolve fix_name=%s route=%s fir_hint=%s",
                fix_name, route, fir_hint
            )
            return None, None, None, None

        fix_meta = WPT_META[fix_idx]
        fix_lat, fix_lon = WPT_COORDS[fix_idx]

        # 1) Compute actual geometric point from anchor fix
        dest_lat, dest_lon = _destination_point(
            fix_lat,
            fix_lon,
            point["bearing"],
            point["dist_nm"]
        )

        # ---------------------------------------------------------
        # MODE 1: opposite side is a direct waypoint
        #
        # POLICY:
        # Include every touched leg fully, but do not include extra untouched legs.
        # ---------------------------------------------------------
        if other_point and other_point.get("type") == "waypoint":
            other_name = (
                other_point.get("display_name")
                or other_point.get("name")
                or ""
            ).strip().upper()

            snapped = _resolve_dist_dir_against_waypoint_on_route(
                route=route,
                anchor_name=fix_name,
                target_dist_nm=point["dist_nm"],
                bearing_deg=point["bearing"],
                fixed_name=other_name,
            )

            if snapped:
                log.info(
                    "✅ MODE1 SNAP: raw=%s route=%s anchor=%s fixed=%s target=%.2fNM -> snapped=%s (%s)",
                    point.get("raw"),
                    route,
                    fix_name,
                    other_name,
                    point["dist_nm"],
                    snapped["name"],
                    snapped.get("snap_reason", "")
                )
                return snapped, dest_lat, dest_lon, (fix_lat, fix_lon)

            # fallback to old simple special case if route-leg resolution could not decide
            first_meta, first_lat, first_lon, first_dist_nm = _first_waypoint_in_direction_on_airway(
                route=route,
                anchor_name=fix_name,
                bearing_deg=point["bearing"]
            )

            if first_meta is not None:
                first_name = first_meta["name"].strip().upper()

                if first_name == other_name and point["dist_nm"] < first_dist_nm:
                    snapped = {
                        **fix_meta,
                        "distance_nm": round(point["dist_nm"], 2),
                        "snap_reason": "first airway waypoint is fixed endpoint -> snap to anchor"
                    }

                    log.info(
                        "✅ ROUTE LEG SNAP (fallback): raw=%s route=%s anchor=%s fixed=%s target=%.2fNM first=%s first_dist=%.2fNM -> snapped=%s",
                        point.get("raw"),
                        route,
                        fix_name,
                        other_name,
                        point["dist_nm"],
                        first_name,
                        first_dist_nm,
                        snapped["name"]
                    )

                    return snapped, dest_lat, dest_lon, (fix_lat, fix_lon)


       # ---------------------------------------------------------
        # MODE 2: opposite side is also dist_dir
        # Example:
        #   165KM WEST OF BIKNO - 30KM WEST OF AKLAS
        #
        # Rule:
        #   Resolve THIS endpoint independently from its own fix_name,
        #   compute the geometric point, then snap to the NEAREST
        #   waypoint on the SAME ROUTE.(didnt delete it)
        # ---------------------------------------------------------
        # --------------------------------------------------------
        # if other_point and other_point.get("type") == "dist_dir":
        # snapped = _nearest_waypoint_on_route(
        # route=route,
        # lat=dest_lat,
        # lon=dest_lon,
        # exclude_name=""
        # )

        # if snapped:
        # snapped["snap_reason"] = "dist_dir->dist_dir nearest waypoint on same route"
        # log.info(
        #     "dist_dir independent-snap: raw=%s route=%s anchor=%s dest=(%.5f,%.5f) -> snapped=%s (%.2fNM)",
        #     point.get("raw"),
        #     route,
        #     fix_name,
        #     dest_lat,
        #     dest_lon,
        # snapped["name"],
        # snapped["distance_nm"]
        # )
        # return snapped, dest_lat, dest_lon, (fix_lat, fix_lon)
        # ----------------------------------------------------------


        # ---------------------------------------------------------
        # MODE 3: fallback airway walk
        # ---------------------------------------------------------
        best = _resolve_dist_dir_on_airway(
            airway=route,
            anchor_name=fix_name,
            anchor_lat=fix_lat,
            anchor_lon=fix_lon,
            target_dist_nm=point["dist_nm"],
            bearing_deg=point["bearing"]
        )

        return best, dest_lat, dest_lon, (fix_lat, fix_lon)

    except Exception as e:
        log.exception(
            "resolve_dist_dir_point failed: route=%s point=%s other_point=%s",
            route, point, other_point
        )
        return None, None, None, None

# ── Distance/Direction constants & regex ─────────────────────────────────────
DIR_BEARING = {
    'NORTH': 0,   'NORTHEAST': 45,  'EAST': 90,   'SOUTHEAST': 135,
    'SOUTH': 180, 'SOUTHWEST': 225, 'WEST': 270,  'NORTHWEST': 315,
    'N': 0,  'NE': 45,  'E': 90,  'SE': 135,
    'S': 180, 'SW': 225, 'W': 270, 'NW': 315,
}
KM_TO_NM = 0.539957


DIST_DIR_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(KM|NM|NAUTICAL\s+MILES?)\s+'
    r'(NORTH(?:EAST|WEST)?|SOUTH(?:EAST|WEST)?|EAST|WEST|NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST|[NSEW]{1,2})\s+'
    r'OF\s+'
    r'([A-Z][A-Z0-9\s]{1,20}?)'
    r'(?:\s+VOR(?:/DME)?\s*\'?\s*([A-Z]{2,4})\'?)?'
    r'(?=\s*[-–.]|$)',
    re.IGNORECASE
)



_DIST_DIR_TOKEN = (
    r'(?:\d+(?:\.\d+)?[ ]*(?:KM|NM|NAUTICAL[ ]+MILES?)'
    r'[ ]+(?:NORTH(?:EAST|WEST)?|SOUTH(?:EAST|WEST)?|EAST|WEST|NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST|[NSEW]{1,2})'
    r'[ ]+OF[ ]+[A-Z][A-Z0-9 ]{1,20}'
    r'(?:[ ]+VOR(?:/DME)?[ ]*\'?[ ]*[A-Z]{2,4}\'?)?)'
)


_COORD_TOKEN = (
    r'(?:[NS]\d{6}[EW]\d{7}|[NS]\d{6}[EW]\d{6}|\d{6}[NS]\d{7}[EW]'
    r'|[NS]\d{4}[EW]\d{5}|\d{4}[NS]\d{5}[EW]|\d{4}[NS]\d{4}[EW])'
)



_VOR_TOKEN = (
    r"[A-Z][A-Z0-9\s]{1,40}?"
    r"\s+VOR(?:/DME)?\s*'?\s*[A-Z]{2,4}\s*'?"
)

_WPT_TOKEN = r"[A-Z][A-Z0-9 ']*[A-Z0-9']"

_POINT = f'(?:{_COORD_TOKEN}|{_DIST_DIR_TOKEN}|{_VOR_TOKEN}|{_WPT_TOKEN})'

ROUTE_SEG_RE = re.compile(
    r'(?:\b\d+\.\s*)?([A-Z0-9]\d{1,4})\s*:\s*'
    r'(' + _POINT + r')\s*[-–]\s*(' + _POINT + r')',
    re.IGNORECASE,
)

SEG_OF_RTE_RE = re.compile(
    r'SEGMENT\s+(' + _POINT + r')\s*[-–]\s*(' + _POINT + r')'
    r'(?=\s+OF\s+ATS\s+RTE\s+[A-Z0-9]{1,5})'
    r'\s+OF\s+ATS\s+RTE\s+([A-Z0-9]{1,5})',
    re.IGNORECASE,
)


ROUTE_ONLY_RE = re.compile(
    r'^\s*(?:\d+\.\s*)?([A-Z0-9]\d{1,4})\s*:?\s*\.\s*$',
    re.IGNORECASE | re.MULTILINE
)

STANDALONE_COORD_RE = re.compile(_COORD_TOKEN, re.IGNORECASE)


def sanitize_text(text: str) -> str:
    text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
    return text[:8000]

def _make_point(token: str) -> dict:
    token = re.sub(r'\s+', ' ', token.strip().upper())
    token = token.rstrip('.').strip()

    # Distance/direction pattern first
    m = DIST_DIR_RE.search(token)
    if m:
        raw_dist = float(m.group(1))
        unit     = m.group(2).upper().replace(" ", "")
        dir_str  = m.group(3).upper()
        fix_name = m.group(5).strip().upper() if m.group(5) else m.group(4).strip().upper()
        fix_name = re.sub(r'\s+VOR(?:/DME)?.*$', '', fix_name).strip()

        bearing  = DIR_BEARING.get(dir_str)
        dist_nm  = round(raw_dist * KM_TO_NM, 2) if unit == 'KM' else raw_dist
        dist_km  = raw_dist if unit == 'KM' else round(raw_dist / KM_TO_NM, 2)

        if bearing is not None:
            return {
                "type": "dist_dir",
                "raw": token,
                "fix_name": fix_name,
                "dist_km": dist_km,
                "dist_nm": dist_nm,
                "unit": unit,
                "bearing": bearing,
                "lat": None,
                "lon": None
            }

    # VOR identifier extraction from full token
    vor_m = re.search(r"\bVOR(?:/DME)?\s*'?\s*([A-Z]{2,4})'?\s*$", token)
    lookup = vor_m.group(1).strip().upper() if vor_m else token.strip(" '")

    # Raw coordinate
    coord = parse_coord(lookup)
    if coord:
        return {
            "type": "coord",
            "raw": token,
            "orig_lat": coord["lat"],
            "orig_lon": coord["lon"],
            **coord
        }

    # Named waypoint
    return {"type": "waypoint", "name": lookup}

def extract_segments(notam_text: str) -> list[dict]:
    segments = []
    seen = set()

    def _clean_seg_point(token: str) -> str:
        if not token:
            return ""

        token = token.strip().upper()

        # remove route/control trailing text accidentally swallowed into the point
        token = re.sub(r'\s+OF\s+ATS\s+RTE\s+[A-Z0-9]+.*$', '', token, flags=re.IGNORECASE)
        token = re.sub(r'\s+CLSD.*$', '', token, flags=re.IGNORECASE)
        token = re.sub(r'\s+BTN.*$', '', token, flags=re.IGNORECASE)
        token = re.sub(r'\s+FROM.*$', '', token, flags=re.IGNORECASE)
        token = re.sub(r'\s+AT\s+.*$', '', token, flags=re.IGNORECASE)

        return token.strip(" .:-")

    def add_seg(route, raw_a, raw_b, raw_text=""):
        route = (route or "").strip().upper()
        raw_a = _clean_seg_point(raw_a)
        raw_b = _clean_seg_point(raw_b)

        if not route or not raw_a or not raw_b:
            return

        key = f"{route}:{raw_a.upper()}:{raw_b.upper()}"
        if key in seen:
            return

        seen.add(key)
        segments.append({
            "route": route,
            "raw": raw_text,
            "point_a": _make_point(raw_a),
            "point_b": _make_point(raw_b),
        })

    for m in ROUTE_SEG_RE.finditer(notam_text):
        add_seg(m.group(1), m.group(2), m.group(3), m.group(0))

    for m in SEG_OF_RTE_RE.finditer(notam_text):
        route = m.group(3).strip().upper() if m.group(3) else "UNK"
        if route in ("OF", "ATS", "RTE", "AND", "BTN", "CLR"):
            continue

        raw_a = m.group(1)
        raw_b = m.group(2)

        # ✅ important for:
        # SEGMENT N350737E1000535 - OMBON OF ATS RTE Y1 CLSD AT ...
        raw_a = _clean_seg_point(raw_a)
        raw_b = _clean_seg_point(raw_b)

        add_seg(route, raw_a, raw_b, m.group(0))

    if not segments:
        for line in notam_text.splitlines():
            line_up = line.strip().upper()
            coords = STANDALONE_COORD_RE.findall(line_up)
            wpts = [
                w for w in re.findall(r'\b([A-Z]{3,6})\b', line_up)
                if w not in {
                    "NOTAM", "CLSD", "FROM", "WITHIN", "SEGMENT", "ATS",
                    "RTE", "BTN", "AND", "FLW", "INCL", "INCLUSIVE", "BELOW", "ABOVE"
                }
            ]
            if coords and wpts:
                add_seg("UNK", coords[0], wpts[0], line)
            elif len(coords) >= 2:
                add_seg("UNK", coords[0], coords[1], line)

    return segments


def extract_route_only_segments(notam_text: str, fir_hint: str) -> list[dict]:
    """
    Extract NOTAM lines like:
        W189.
        V116.
    and convert them into FIR-scoped airway closure segments.
    """
    segments = []
    seen = set()

    for line in notam_text.splitlines():
        line_up = line.strip().upper()

        # Skip lines that already define explicit segments
        if "-" in line_up or "SEGMENT" in line_up:
            continue

        m = ROUTE_ONLY_RE.match(line_up)
        if not m:
            continue

        route = m.group(1).strip().upper()
        blocks = _get_airway_blocks_in_fir(route, fir_hint)

        for start_meta, end_meta in blocks:
            key = (
                route,
                start_meta["name"].strip().upper(),
                end_meta["name"].strip().upper()
            )
            if key in seen:
                continue
            seen.add(key)

            log.info(
                "route-only closure: route=%s fir_hint=%s -> block=%s to %s",
                route,
                fir_hint,
                start_meta["name"],
                end_meta["name"]
            )

            segments.append({
                "route": route,
                "raw": line.strip(),
                "point_a": {"type": "waypoint", "name": start_meta["name"].strip().upper()},
                "point_b": {"type": "waypoint", "name": end_meta["name"].strip().upper()},
                "route_only_block": True,
            })

    return segments

# ── Security headers ──────────────────────────────────────────────────────────
@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://unpkg.com; "
        "img-src 'self' data: https://*.tile.openstreetmap.org https://*.basemaps.cartocdn.com https://*.tile.carto.com; "
        "connect-src 'self' https://unpkg.com; font-src 'self' https://fonts.gstatic.com;"
    )
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")



@app.route("/api/health")
def health():
    healthy = _kdtree is not None and len(WPT_NAMES) > 0
    return jsonify({
        "status": "ok" if healthy else "degraded",
        "waypoints_loaded": len(WPT_NAMES),
        "kdtree_ready": _kdtree is not None,
        "airways_loaded": len(WPT_BY_AIRWAY),
    }), 200 if healthy else 503

@app.route("/api/analyze", methods=["POST"])
@rate_limit
def analyze():
    if not request.is_json:
        abort(415)
    body      = request.get_json(silent=True) or {}
    raw_notam = body.get("notam", "")
    if not isinstance(raw_notam, str) or not raw_notam.strip():
        return jsonify({"error": "notam field is required and must be a non-empty string"}), 400


    notam_text = sanitize_text(raw_notam)
    notam_id   = (re.search(r'[A-Z]\d{4}/\d{2}', notam_text) or type('', (), {'group': lambda *a: 'UNKNOWN'})()).group(0)
    a_fir_m = re.search(r'A\)\s*([A-Z]{4})', notam_text)
    q_fir_m = re.search(r'Q\)\s*([A-Z]{4})', notam_text)

    notam_fir_hint = ""
    if a_fir_m:
        notam_fir_hint = a_fir_m.group(1)[:2].upper()
    elif q_fir_m:
        notam_fir_hint = q_fir_m.group(1)[:2].upper()
    fl_m       = re.search(r'FL\s*(\d{3})\s*TO\s*FL\s*(\d{3})', notam_text, re.IGNORECASE)
    fl_range   = f"FL{fl_m.group(1)}–FL{fl_m.group(2)}" if fl_m else None
    valid_b    = re.search(r'B\)(\d{10})', notam_text)
    valid_c    = re.search(r'C\)(\d{10})', notam_text)


    segments = extract_segments(notam_text)
    
    # add route-only airway closures like "W189." or "V116."
    route_only_segments = extract_route_only_segments(notam_text, notam_fir_hint)
    segments.extend(route_only_segments)

    for seg in segments:
        route = seg.get("route", "")


        # ── STEP 1: Resolve named waypoints ───────────────────────────────────
        for key in ("point_a", "point_b"):
            pt = seg[key]
            if pt["type"] != "waypoint":
                continue
            name = pt["name"]
            idx = _pick_waypoint_index(name, route=route, fir_hint=notam_fir_hint)

            if idx is not None:
                meta = WPT_META[idx]
                lat, lon = WPT_COORDS[idx]

                fir_by_dist = get_fir_by_distance(lat, lon, threshold_nm=50.0)
                display_fir = fir_by_dist if fir_by_dist else meta.get("fir", "")

                pt.update({
                    "resolved": meta,
                    "lat": lat,
                    "lon": lon,
                    "display_name": meta["name"],
                    "display_lat": lat,
                    "display_lon": lon,
                    "display_fir": display_fir,
                    "all_airways": [WPT_META[i]["airway"] for i in WPT_AIRWAYS[name]],
                })
            else:
                pt.update({
                    "resolved": None,
                    "display_name": name,
                    "display_lat": None,
                    "display_lon": None,
                    "display_fir": "",
                })

        # ── STEP 2: Resolve dist_dir points independently ───────────────
        for key in("point_a", "point_b"):

            pt = seg[key]

            if pt["type"] != "dist_dir":
                continue

            other_point = seg["point_b" if key == "point_a" else "point_a"]

            resolved = resolve_dist_dir_point(
                route,
                pt,
                other_point=other_point,
                fir_hint=notam_fir_hint
            )

            if not isinstance(resolved, tuple) or len(resolved) != 4:
                log.error(
                    "resolve_dist_dir_point returned invalid value: route=%s point=%s returned=%r",
                    route, pt, resolved
                )
                best, dest_lat, dest_lon, anchor = None, None, None, None
            else:
                best, dest_lat, dest_lon, anchor = resolved


            if anchor is None:
                pt.update({
                    "lat": None,
                    "lon": None,
                    "orig_lat": None,
                    "orig_lon": None,
                    "nearest": None,
                    "display_name": pt["raw"],
                    "display_lat": None,
                    "display_lon": None,
                    "display_fir": "",
                })
                continue

            fix_lat, fix_lon = anchor

            pt["lat"] = dest_lat
            pt["lon"] = dest_lon
            pt["orig_lat"] = dest_lat
            pt["orig_lon"] = dest_lon
            
            # ✅ IMPORTANT:
            # # For dist_dir -> dist_dir, do NOT fallback to generic nearest.
            # # That collapses results to the same anchor (BIKNO / AKLAS).
            if best is None and not (
                other_point and other_point.get("type") == "dist_dir"
            ):
                nearby = nearest_waypoints(dest_lat, dest_lon, k=20)

                scored = []
                for w in nearby:
                    bear_to_w = _bearing(fix_lat, fix_lon, w["lat"], w["lon"])
                    ang_diff = abs(((bear_to_w - pt["bearing"]) + 180) % 360 - 180)
                    dist_from_fix = _dist_nm(fix_lat, fix_lon, w["lat"], w["lon"])

                    if ang_diff <= 90 and dist_from_fix >= pt["dist_nm"] * 0.5:
                        scored.append((dist_from_fix, ang_diff, w))

                if scored:
                    scored.sort(key=lambda x: (x[0], x[1]))
                    best = scored[0][2]
                elif nearby:
                    best = nearby[0]

            if best:
                pt["nearest"] = best
                pt["display_name"] = best["name"]
                pt["display_lat"] = best["lat"]
                pt["display_lon"] = best["lon"]

                log.info(
                    "STEP2 resolved: route=%s raw=%s fix_name=%s -> display=%s snap_reason=%s",
                    route,
                    pt.get("raw"),
                    pt.get("fix_name"),
                    best.get("name"),
                    best.get("snap_reason", "")
                )

                fir = get_fir_by_distance(best["lat"], best["lon"], threshold_nm=50.0)
                pt["display_fir"] = fir if fir else best.get("fir", "")
            else:
                pt["nearest"] = None
                pt["display_name"] = pt["raw"]
                pt["display_lat"] = dest_lat
                pt["display_lon"] = dest_lon
                pt["display_fir"] = ""

        # ── STEP 2B: Special handling for dist_dir -> dist_dir on same route ──
        pa = seg.get("point_a", {})
        pb = seg.get("point_b", {})

        if (
            route
            and pa.get("type") == "dist_dir"
            and pb.get("type") == "dist_dir"
            and route in WPT_BY_AIRWAY
        ):
            leg_a = _locate_dist_dir_leg_on_airway(
                route=route,
                anchor_name=pa.get("fix_name", ""),
                target_dist_nm=pa.get("dist_nm", 0.0),
                bearing_deg=pa.get("bearing", 0.0),
            )
            leg_b = _locate_dist_dir_leg_on_airway(
                route=route,
                anchor_name=pb.get("fix_name", ""),
                target_dist_nm=pb.get("dist_nm", 0.0),
                bearing_deg=pb.get("bearing", 0.0),
            )

            if leg_a and leg_b:
                idxs = WPT_BY_AIRWAY[route]

                # The closure should encompass all intersected legs.
                boundary_min = min(
                    leg_a["from_pos"], leg_a["to_pos"],
                    leg_b["from_pos"], leg_b["to_pos"],
                )
                boundary_max = max(
                    leg_a["from_pos"], leg_a["to_pos"],
                    leg_b["from_pos"], leg_b["to_pos"],
                )

                left_meta = WPT_META[idxs[boundary_min]]
                right_meta = WPT_META[idxs[boundary_max]]

                left_lat, left_lon = WPT_COORDS[idxs[boundary_min]]
                right_lat, right_lon = WPT_COORDS[idxs[boundary_max]]

                # Decide which original endpoint is on which side of the route
                mid_a = (leg_a["from_pos"] + leg_a["to_pos"]) / 2
                mid_b = (leg_b["from_pos"] + leg_b["to_pos"]) / 2

                if mid_a >= mid_b:
                    # point_a is on the higher/right side of route order
                    a_meta, a_lat, a_lon = right_meta, right_lat, right_lon
                    b_meta, b_lat, b_lon = left_meta, left_lat, left_lon
                else:
                    a_meta, a_lat, a_lon = left_meta, left_lat, left_lon
                    b_meta, b_lat, b_lon = right_meta, right_lat, right_lon

                pa["nearest"] = a_meta
                pa["display_name"] = a_meta["name"]
                pa["display_lat"] = a_lat
                pa["display_lon"] = a_lon
                pa["display_fir"] = get_fir_by_distance(a_lat, a_lon, threshold_nm=50.0) or a_meta.get("fir", "")
                pa["snap_reason"] = "dist_dir->dist_dir enclosing airway boundary"

                pb["nearest"] = b_meta
                pb["display_name"] = b_meta["name"]
                pb["display_lat"] = b_lat
                pb["display_lon"] = b_lon
                pb["display_fir"] = get_fir_by_distance(b_lat, b_lon, threshold_nm=50.0) or b_meta.get("fir", "")
                pb["snap_reason"] = "dist_dir->dist_dir enclosing airway boundary"

                log.info(
                    "STEP2B dist_dir->dist_dir enclosure: route=%s raw_a=%s raw_b=%s -> %s to %s",
                    route,
                    pa.get("raw"),
                    pb.get("raw"),
                    pa["display_name"],
                    pb["display_name"],
                )

        # ── STEP 3: Resolve raw coordinate points (route-leg enclosure first) ──
        for key in ("point_a", "point_b"):
            pt = seg[key]
            if pt["type"] != "coord":
                continue

            if pt.get("lat") is None:
                pt.update({
                    "nearest": None,
                    "display_name": pt["raw"],
                    "display_lat": None,
                    "display_lon": None,
                    "display_fir": ""
                })
                continue

            other = seg["point_b" if key == "point_a" else "point_a"]
            other_name = (
                other.get("display_name")
                or other.get("name")
                or ""
            ).strip().upper()

            # ---------------------------------------------------------
            # PRIORITY: if this is coord <-> waypoint on a known route,
            # snap by containing airway leg so the full touched leg is included
            # without adding extra untouched legs.
            # ---------------------------------------------------------
            snapped = None

            if route and other_name and other.get("type") == "waypoint":
                leg = _locate_coord_leg_on_airway(route, pt["lat"], pt["lon"])

                if leg:
                    fixed_positions = _route_positions(route, other_name)
                    if fixed_positions:
                        # choose fixed occurrence closest to this leg
                        other_pos = min(
                            fixed_positions,
                            key=lambda p: min(abs(p - leg["from_pos"]), abs(p - leg["to_pos"]))
                        )

                        # If fixed endpoint is on/after the "to" side of the leg,
                        # closure runs from this coord toward that fixed endpoint,
                        # so choose the "from" side to enclose the full touched leg.
                        if other_pos >= leg["to_pos"]:
                            snapped = leg["from_meta"]
                            snap_reason = "coord inside leg -> enclosing from-side waypoint"
                        # If fixed endpoint is on/before the "from" side,
                        # choose the "to" side.
                        elif other_pos <= leg["from_pos"]:
                            snapped = leg["to_meta"]
                            snap_reason = "coord inside leg -> enclosing to-side waypoint"
                        else:
                            # rare ambiguous case: fixed lies within same leg span,
                            # choose nearer endpoint
                            d_from = _dist_nm(pt["lat"], pt["lon"], leg["from_meta"]["lat"], leg["from_meta"]["lon"])
                            d_to = _dist_nm(pt["lat"], pt["lon"], leg["to_meta"]["lat"], leg["to_meta"]["lon"])
                            snapped = leg["from_meta"] if d_from <= d_to else leg["to_meta"]
                            snap_reason = "coord inside leg -> ambiguous fixed position, chose nearer leg endpoint"

                        log.info(
                            "STEP3 route-leg snap: route=%s coord=%s fixed=%s leg=%s-%s detour=%.2fNM -> snapped=%s (%s)",
                            route,
                            pt["raw"],
                            other_name,
                            leg["from_meta"]["name"],
                            leg["to_meta"]["name"],
                            leg["detour_nm"],
                            snapped["name"],
                            snap_reason,
                        )

            # ---------------------------------------------------------
            # Fallback: old nearest/directional logic
            # ---------------------------------------------------------
            if not snapped:
                anchor_lat = other.get("lat") or other.get("display_lat")
                anchor_lon = other.get("lon") or other.get("display_lon")
                anchor_nm  = other.get("display_name") or other.get("name", "")

                nearby = nearest_waypoints(pt["lat"], pt["lon"], k=20)
                pt["nearest_waypoints"] = nearby[:5]

                best = None
                if anchor_lat is not None and nearby:
                    target_brg = _bearing(anchor_lat, anchor_lon, pt["lat"], pt["lon"])
                    anchor_to_coord = _dist_nm(anchor_lat, anchor_lon, pt["lat"], pt["lon"])
                    scored = []
                    for w in nearby:
                        if w["name"].strip().upper() == anchor_nm.strip().upper():
                            continue
                        w_brg = _bearing(anchor_lat, anchor_lon, w["lat"], w["lon"])
                        ang_diff = abs(((w_brg - target_brg) + 180) % 360 - 180)
                        same_route = w.get("airway") == route
                        dir_ok = ang_diff < 90
                        a_to_w = _dist_nm(anchor_lat, anchor_lon, w["lat"], w["lon"])
                        ahead_delta = a_to_w - anchor_to_coord
                        ahead_ok = ahead_delta >= -5.0
                        scored.append({
                            "wpt": w,
                            "angle_diff": ang_diff,
                            "distance_nm": w["distance_nm"],
                            "same_route": same_route,
                            "direction_ok": dir_ok,
                            "ahead_ok": ahead_ok,
                            "ahead_delta": ahead_delta
                        })

                    tiers = [
                        (lambda c: c["ahead_ok"] and c["direction_ok"] and c["same_route"], True),
                        (lambda c: c["ahead_ok"] and c["direction_ok"], True),
                        (lambda c: c["direction_ok"] and c["same_route"], False),
                        (lambda c: c["direction_ok"], False),
                        (lambda c: c["same_route"], False),
                        (lambda c: True, False),
                    ]

                    choice = None
                    for cond, use_ahead in tiers:
                        pool = [c for c in scored if cond(c)]
                        if not pool:
                            continue
                        pool.sort(
                            key=lambda c: (c["ahead_delta"], c["distance_nm"])
                            if use_ahead else
                            (c["distance_nm"], c["angle_diff"])
                        )
                        choice = pool[0]
                        break

                    best = choice["wpt"] if choice else None
                else:
                    best = nearby[0] if nearby else None

                snapped = best

            # ---------------------------------------------------------
            # Apply result
            # ---------------------------------------------------------
            if snapped:
                pt.update({
                    "nearest": snapped,
                    "display_name": snapped["name"],
                    "display_lat": snapped["lat"],
                    "display_lon": snapped["lon"],
                })
                fir_by_dist = get_fir_by_distance(snapped["lat"], snapped["lon"], threshold_nm=50.0)
                pt.update({
                    "display_fir": fir_by_dist if fir_by_dist else snapped.get("fir", "")
                })
            else:
                pt.update({
                    "nearest": None,
                    "display_name": pt["raw"],
                    "display_lat": pt.get("lat"),
                    "display_lon": pt.get("lon"),
                    "display_fir": ""
                })


    result = {
        "notam_id":       notam_id,
        "fl_range":       fl_range,
        "valid_from":     valid_b.group(1) if valid_b else None,
        "valid_to":       valid_c.group(1) if valid_c else None,
        "segments":       segments,
        "total_segments": len(segments),
    }
    log.info("Analyzed NOTAM %s — %d segments", notam_id, len(segments))
    return jsonify(result)


@app.route("/api/nearest", methods=["POST"])
@rate_limit
def nearest():
    if not request.is_json:
        abort(415)
    body = request.get_json(silent=True) or {}
    if "coord" in body:
        coord = parse_coord(str(body["coord"]))
        if not coord:
            return jsonify({"error": "Could not parse coordinate format"}), 400
        lat, lon = coord["lat"], coord["lon"]
    elif "lat" in body and "lon" in body:
        try:
            lat = float(body["lat"]); lon = float(body["lon"])
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid lat/lon"}), 400
    else:
        return jsonify({"error": "Provide coord or lat+lon"}), 400
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return jsonify({"error": "lat/lon out of range"}), 400
    return jsonify({"lat": lat, "lon": lon, "nearest": nearest_waypoints(lat, lon, k=5)})


@app.route("/api/debug/nearest")
def debug_nearest():
    if os.environ.get("FLASK_DEBUG", "false").lower() != "true":
        abort(404)
    coord_str = request.args.get("coord")
    if coord_str:
        parsed = parse_coord(coord_str)
        if not parsed:
            return jsonify({"error": "Cannot parse coord"}), 400
        lat, lon = parsed["lat"], parsed["lon"]
    else:
        try:
            lat = float(request.args["lat"]); lon = float(request.args["lon"])
        except (KeyError, ValueError):
            return jsonify({"error": "Provide coord or lat+lon"}), 400
    return jsonify({"query_lat": lat, "query_lon": lon,
                    "total_wpts_loaded": len(WPT_NAMES), "nearest_10": nearest_waypoints(lat, lon, k=10)})


@app.route("/api/debug/lookup")
def debug_lookup():
    if os.environ.get("FLASK_DEBUG", "false").lower() != "true":
        abort(404)
    name = request.args.get("name", "").strip().upper()
    if not name:
        return jsonify({"error": "Provide name param"}), 400
    idxs = WPT_AIRWAYS.get(name, [])
    if not idxs:
        return jsonify({"found": False, "name": name, "total_wpts_loaded": len(WPT_NAMES)})
    return jsonify({"found": True, "name": name, "entries": [WPT_META[i] for i in idxs]})


@app.errorhandler(404)
def not_found(e): return jsonify({"error": "Not found"}), 404
@app.errorhandler(405)
def method_not_allowed(e): return jsonify({"error": "Method not allowed"}), 405
@app.errorhandler(413)
def too_large(e): return jsonify({"error": "Request too large"}), 413
@app.errorhandler(429)
def rate_limited(e): return jsonify({"error": "Rate limit exceeded"}), 429
@app.errorhandler(500)
def server_error(e):
    log.error("Internal error: %s", e)
    return jsonify({"error": "Internal server error"}), 500


# ── Startup ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DEFAULT_WAYPOINT_PATH = Path("data/wavepoints.csv")
    DATA_PATH = os.environ.get("WAYPOINT_CSV", str(DEFAULT_WAYPOINT_PATH))

    count = load_waypoints(DATA_PATH)

    if count == 0:
        log.warning("Waypoint dataset could not be loaded. App is starting in degraded mode.")

    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=debug)
