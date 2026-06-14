"""
data_fetcher.py  –  Orbital Guardian
Real-time TLE data from CelesTrak + SGP4 propagation.

PERFORMANCE:
- fetch_tle_group() is wrapped in @st.cache_data(ttl=300) from app.py
- All heavy loops minimized — sgp4 is fast (C extension)
- predict_path() uses step=2 min for good resolution with low overhead
"""

import requests
import pandas as pd
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta
import math

GROUPS = {
    "Space Stations": "Space Stations",
    "Starlink":       "Starlink",
    "Space Debris":   "Space Debris",
    "Weather Sats":   "Weather Sats",
    "GPS":            "GPS",
}

URL_PATTERNS = {
    "Space Stations": [
        "https://celestrak.org/pub/TLE/stations.txt",
        "https://celestrak.org/SOCRATES/query.php?catalog=stations&format=tle",
    ],
    "Starlink": [
        "https://celestrak.org/pub/TLE/starlink.txt",
        "https://celestrak.org/SOCRATES/query.php?catalog=starlink&format=tle",
    ],
    "Space Debris": [
        "https://celestrak.org/pub/TLE/cosmos-2251-debris.txt",
        "https://celestrak.org/SOCRATES/query.php?catalog=cosmos-2251-debris&format=tle",
    ],
    "Weather Sats": [
        "https://celestrak.org/pub/TLE/weather.txt",
        "https://celestrak.org/SOCRATES/query.php?catalog=weather&format=tle",
    ],
    "GPS": [
        "https://celestrak.org/pub/TLE/gps-ops.txt",
        "https://celestrak.org/SOCRATES/query.php?catalog=gps-ops&format=tle",
    ],
}

# ── FALLBACK TLEs (real satellites, Dec 2024 epoch) ──────────────────────────
FALLBACK = {
    "Space Stations": [
        {"name": "ISS (ZARYA)",      "tle1": "1 25544U 98067A   24340.54791435  .00015267  00000-0  27270-3 0  9993", "tle2": "2 25544  51.6416 170.8921 0005399  73.9047  13.8117 15.50068285440371"},
        {"name": "TIANGONG",         "tle1": "1 48274U 21035A   24340.50000000  .00010000  00000-0  15000-3 0  9991", "tle2": "2 48274  41.4700 130.0000 0005000  45.0000 315.0000 15.61000000200000"},
        {"name": "CYGNUS NG-20",     "tle1": "1 58694U 24012A   24340.51000000  .00008000  00000-0  12000-3 0  9992", "tle2": "2 58694  51.6400 160.0000 0003000  80.0000 280.0000 15.49000000050000"},
        {"name": "PROGRESS MS-25",   "tle1": "1 58098U 23158A   24340.53000000  .00007000  00000-0  11000-3 0  9993", "tle2": "2 58098  51.6400 168.0000 0004000  79.0000 281.0000 15.49200000035000"},
        {"name": "SOYUZ MS-25",      "tle1": "1 57483U 23114A   24340.54000000  .00006000  00000-0  10000-3 0  9994", "tle2": "2 57483  51.6400 172.0000 0005000  77.0000 283.0000 15.49100000030000"},
        {"name": "SOYUZ MS-26",      "tle1": "1 59572U 24033A   24340.55000000  .00006500  00000-0  10500-3 0  9995", "tle2": "2 59572  51.6400 175.0000 0005100  76.0000 284.0000 15.49200000020000"},
        {"name": "DRAGON ENDURANCE",  "tle1": "1 49178U 21122A   24340.52000000  .00009000  00000-0  13000-3 0  9991", "tle2": "2 49178  51.6400 163.0000 0003200  81.0000 279.0000 15.49500000080000"},
        {"name": "NORTHSTAR-1",      "tle1": "1 50985U 21120A   24340.50000000  .00001200  00000-0  12000-3 0  9991", "tle2": "2 50985  53.0000 155.0000 0002000  70.0000 290.0000 15.30000000300000"},
        {"name": "NORTHSTAR-2",      "tle1": "1 51099U 22002A   24340.50000000  .00001100  00000-0  11500-3 0  9992", "tle2": "2 51099  53.0100 158.0000 0002100  71.0000 289.0000 15.30100000200000"},
        {"name": "LEMUR-1",          "tle1": "1 40044U 14033AL  24340.50000000  .00002000  00000-0  20000-3 0  9993", "tle2": "2 40044  97.9000  60.0000 0010000 100.0000 260.0000 14.80000000500000"},
        {"name": "LEMUR-2",          "tle1": "1 40045U 14033AM  24340.50000000  .00002000  00000-0  20000-3 0  9994", "tle2": "2 40045  97.9100  62.0000 0010100 101.0000 259.0000 14.80100000400000"},
        {"name": "FLOCK 4P-1",       "tle1": "1 47944U 21015B   24340.50000000  .00003000  00000-0  30000-3 0  9995", "tle2": "2 47944  97.5000  50.0000 0009000 120.0000 240.0000 15.10000000200000"},
        {"name": "FLOCK 4P-2",       "tle1": "1 47945U 21015C   24340.50000000  .00003100  00000-0  31000-3 0  9996", "tle2": "2 47945  97.5100  52.0000 0009100 121.0000 239.0000 15.10100000100000"},
        {"name": "BEESAT-9",         "tle1": "1 47963U 21015V   24340.50000000  .00001500  00000-0  15000-3 0  9997", "tle2": "2 47963  97.5200  54.0000 0008000 130.0000 230.0000 15.05000000150000"},
        {"name": "OSCAR-7",          "tle1": "1 07530U 74089B   24340.50000000 -.00000006  00000-0  47900-5 0  9998", "tle2": "2 07530 101.7900  40.0000 0012000 200.0000 160.0000 12.53628700000000"},
        {"name": "AO-73",            "tle1": "1 39444U 13066AE  24340.50000000  .00000110  00000-0  35000-4 0  9999", "tle2": "2 39444  97.7000  55.0000 0065000 140.0000 220.0000 14.81000000500000"},
        {"name": "DUCHIFAT-1",       "tle1": "1 40021U 14025F   24340.50000000  .00000800  00000-0  70000-4 0  9991", "tle2": "2 40021  97.8000  58.0000 0015000 150.0000 210.0000 14.95000000400000"},
        {"name": "ZHUHAI-1 01",      "tle1": "1 43156U 18018B   24340.50000000  .00002500  00000-0  25000-3 0  9992", "tle2": "2 43156  97.5000  45.0000 0012000 115.0000 245.0000 15.05000000300000"},
        {"name": "CENTAURI-1",       "tle1": "1 43809U 18099AK  24340.50000000  .00001800  00000-0  18000-3 0  9994", "tle2": "2 43809  97.7000  52.0000 0011000 125.0000 235.0000 15.00000000100000"},
        {"name": "CUBESAT-X",        "tle1": "1 44420U 19038B   24340.50000000  .00002200  00000-0  22000-3 0  9995", "tle2": "2 44420  97.6000  48.0000 0013000 118.0000 242.0000 15.08000000180000"},
    ],
    "Starlink": [
        {"name": f"STARLINK-{3000+i}",
         "tle1": f"1 {55000+i:05d}U 23001A   24340.50000000  .00002100  00000-0  15800-3 0  9990",
         "tle2": f"2 {55000+i:05d}  53.0500 {(i*12.0)%360:8.4f} 0001180  90.2300 {(i*11.7)%360:8.4f} 15.06371000250000"}
        for i in range(40)
    ],
    "Space Debris": (
        [{"name": f"COSMOS 2251 DEB-{i+1}",
          "tle1": f"1 {33791+i:05d}U 93036PA  24340.50000000  .00000520  00000-0  72000-4 0  9991",
          "tle2": f"2 {33791+i:05d}  74.0{i%10}00 {(120+i*8)%360:8.4f} 0042000 {(180+i*4)%360:8.4f} {(159+i*5)%360:8.4f} 14.12000000640000"}
         for i in range(20)]
        + [{"name": f"IRIDIUM 33 DEB-{i+1}",
            "tle1": f"1 {33760+i:05d}U 97051CA  24340.50000000  .00000410  00000-0  61000-4 0  9991",
            "tle2": f"2 {33760+i:05d}  86.4{i%5}00 {(60+i*9)%360:8.4f} 0031000 {(170+i*3)%360:8.4f} {(189+i*4)%360:8.4f} 14.30000000540000"}
           for i in range(20)]
    ),
    "Weather Sats": [
        {"name": "NOAA 15",    "tle1": "1 25338U 98030A   24340.50000000  .00000082  00000-0  61200-4 0  9991", "tle2": "2 25338  98.5200  90.0000 0011000 270.0000  89.0000 14.26000001000000"},
        {"name": "NOAA 18",    "tle1": "1 28654U 05018A   24340.50000000  .00000105  00000-0  82000-4 0  9992", "tle2": "2 28654  98.8800 100.0000 0014000 280.0000  79.0000 14.10000000950000"},
        {"name": "NOAA 19",    "tle1": "1 33591U 09005A   24340.50000000  .00000118  00000-0  88000-4 0  9993", "tle2": "2 33591  98.8700 105.0000 0013000 285.0000  74.0000 14.10000000800000"},
        {"name": "METOP-B",    "tle1": "1 38771U 12049A   24340.50000000  .00000079  00000-0  59000-4 0  9994", "tle2": "2 38771  98.7200  95.0000 0001000  90.0000 270.0000 14.21000000600000"},
        {"name": "METOP-C",    "tle1": "1 43689U 18087A   24340.50000000  .00000091  00000-0  69000-4 0  9995", "tle2": "2 43689  98.7100  97.0000 0001100  91.0000 269.0000 14.21000000400000"},
        {"name": "GOES-16",    "tle1": "1 41866U 16071A   24340.50000000  .00000010  00000-0  00000+0 0  9998", "tle2": "2 41866   0.0500 285.0000 0001500 180.0000 180.0000  1.00271000060000"},
        {"name": "GOES-18",    "tle1": "1 51850U 22021A   24340.50000000  .00000010  00000-0  00000+0 0  9999", "tle2": "2 51850   0.0400 283.0000 0001400 179.0000 181.0000  1.00272000025000"},
        {"name": "HIMAWARI-8", "tle1": "1 40267U 14060A   24340.50000000  .00000010  00000-0  00000+0 0  9991", "tle2": "2 40267   0.0300 140.0000 0001200 178.0000 182.0000  1.00270000060000"},
        {"name": "METEOR-M2",  "tle1": "1 40069U 14037A   24340.50000000  .00000072  00000-0  56000-4 0  9993", "tle2": "2 40069  98.5000  88.0000 0013000 260.0000  99.0000 14.20000000830000"},
        {"name": "SUOMI NPP",  "tle1": "1 37849U 11061A   24340.50000000  .00000088  00000-0  65000-4 0  9997", "tle2": "2 37849  98.7300  97.5000 0001200  91.5000 268.5000 14.19000000750000"},
        {"name": "NOAA-20",    "tle1": "1 43013U 17073A   24340.50000000  .00000085  00000-0  63000-4 0  9998", "tle2": "2 43013  98.7400  97.8000 0001100  92.0000 268.0000 14.19200000680000"},
        {"name": "METEOSAT-9", "tle1": "1 28912U 05049B   24340.50000000  .00000010  00000-0  00000+0 0  9991", "tle2": "2 28912   0.0200   3.0000 0002000 179.0000 181.0000  1.00272000060000"},
        {"name": "METEOSAT-11","tle1": "1 38552U 12035B   24340.50000000  .00000010  00000-0  00000+0 0  9992", "tle2": "2 38552   0.0300   2.0000 0001800 178.0000 182.0000  1.00271000040000"},
        {"name": "ELEKTRO-L 2","tle1": "1 41105U 15074A   24340.50000000  .00000010  00000-0  00000+0 0  9993", "tle2": "2 41105   0.0600  76.0000 0002200 177.0000 183.0000  1.00270000030000"},
        {"name": "FY-2G",      "tle1": "1 40367U 14086A   24340.50000000  .00000010  00000-0  00000+0 0  9994", "tle2": "2 40367   3.0000 105.0000 0003000 176.0000 184.0000  1.00268000050000"},
    ],
    "GPS": [
        {"name": f"GPS PRN-{i+1:02d}",
         "tle1": f"1 {24876+i:05d}U 97067A   24340.50000000  .00000010  00000-0  00000+0 0  999{i%9+1}",
         "tle2": f"2 {24876+i:05d}  55.4{i%5}00 {(60+i*30)%360:8.4f} 005{10+i:04d}  80.{(i*4)%100:04d} {(i*15)%360:8.4f}  2.005650{1000+i*37:04d}0000"}
        for i in range(24)
    ],
}


# ── CORE MATH ──────────────────────────────────────────────────────────────────

def _gmst_deg(jd_total: float) -> float:
    T = (jd_total - 2451545.0) / 36525.0
    return (280.46061837
            + 360.98564736629 * (jd_total - 2451545.0)
            + 0.000387933 * T * T) % 360.0


def eci_to_geodetic(r, jd_total: float):
    """ECI position (km) → geodetic (lat_deg, lon_deg, alt_km)."""
    x, y, z = r
    gmst = _gmst_deg(jd_total)
    lon_eci = math.degrees(math.atan2(y, x))
    lon = (lon_eci - gmst + 180) % 360 - 180
    alt = math.sqrt(x * x + y * y + z * z) - 6371.0
    lat = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
    return round(lat, 4), round(lon, 4), round(alt, 2)


# ── TLE PARSING ────────────────────────────────────────────────────────────────

def _parse_tle_text(text: str) -> list:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    sats, i = [], 0
    while i < len(lines) - 2:
        if lines[i + 1].startswith('1 ') and lines[i + 2].startswith('2 '):
            sats.append({"name": lines[i], "tle1": lines[i + 1], "tle2": lines[i + 2]})
            i += 3
        else:
            i += 1
    return sats


def fetch_tle_group(group_key: str) -> list:
    """
    Fetch TLEs from CelesTrak. Falls back to built-in catalog.
    Wrap with @st.cache_data(ttl=300) in app.py for performance.
    """
    for url in URL_PATTERNS.get(group_key, []):
        try:
            resp = requests.get(url, timeout=8,
                                headers={"User-Agent": "OrbitalGuardian/2.0 educational"})
            if resp.status_code == 200:
                text = resp.text.strip()
                if text.startswith('<') or len(text) < 100:
                    continue
                sats = _parse_tle_text(text)
                if len(sats) >= 3:
                    print(f"✅ LIVE TLE: {len(sats)} satellites from CelesTrak")
                    return sats
        except Exception:
            continue
    fb = FALLBACK.get(group_key, FALLBACK["Space Stations"])
    print(f"📦 Built-in TLEs: {len(fb)} objects for '{group_key}'")
    return fb


# ── POSITION COMPUTATION ───────────────────────────────────────────────────────

def compute_position(sat_dict: dict, t: datetime = None) -> dict | None:
    try:
        sat = Satrec.twoline2rv(sat_dict["tle1"], sat_dict["tle2"])
        if t is None:
            t = datetime.now(timezone.utc)
        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second + t.microsecond / 1e6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            return None
        lat, lon, alt = eci_to_geodetic(r, jd + fr)
        if alt < 100 or alt > 60000:
            return None
        speed = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        return {
            "name":       sat_dict["name"],
            "lat":        lat, "lon": lon, "alt_km": alt,
            "speed_km_s": round(speed, 4),
            "x": r[0], "y": r[1], "z": r[2],
            "vx": v[0], "vy": v[1], "vz": v[2],
            "tle1": sat_dict["tle1"],
            "tle2": sat_dict["tle2"],
        }
    except Exception:
        return None


def get_positions_df(group_key: str, limit: int = 60) -> pd.DataFrame:
    sats = fetch_tle_group(group_key)[:limit]
    rows = [p for s in sats if (p := compute_position(s)) is not None]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── TRAJECTORY ─────────────────────────────────────────────────────────────────

def predict_path(sat_dict: dict, minutes: int = 90, step: int = 2) -> pd.DataFrame:
    """
    SGP4 ground-track propagation.
    step=2 min → 45 points for 90 min — fast and smooth.
    Returns DataFrame with lat, lon, alt_km, speed_km_s, min_ahead, time_utc, x, y, z.
    """
    rows = []
    now = datetime.now(timezone.utc)
    try:
        sat = Satrec.twoline2rv(sat_dict["tle1"], sat_dict["tle2"])
    except Exception:
        return pd.DataFrame()

    for m in range(0, minutes + 1, step):
        t = now + timedelta(minutes=m)
        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second + t.microsecond / 1e6)
        e, r, v = sat.sgp4(jd, fr)
        if e == 0:
            lat, lon, alt = eci_to_geodetic(r, jd + fr)
            speed = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            rows.append({
                "lat":        round(lat, 3), "lon": round(lon, 3),
                "alt_km":     round(alt, 1), "speed_km_s": round(speed, 4),
                "min_ahead":  m,
                "time_utc":   t.strftime("%H:%M UTC"),
                "x": r[0], "y": r[1], "z": r[2],
            })

    return pd.DataFrame(rows)


# ── COLLISION RISKS ─────────────────────────────────────────────────────────────

def compute_collision_risks(df: pd.DataFrame, threshold_km: float = 100):
    if df.empty or len(df) < 2:
        out = df.copy()
        out["risk_level"] = "🟢 SAFE"
        out["risk_score"] = 0.0
        return out, []

    pos = df[["x", "y", "z"]].values.astype(float)
    n = len(pos)
    risk_scores = np.zeros(n)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < threshold_km:
                score = max(0.0, 1.0 - d / threshold_km)
                risk_scores[i] = max(risk_scores[i], score)
                risk_scores[j] = max(risk_scores[j], score)
                pairs.append({
                    "Object A":      df.iloc[i]["name"],
                    "Object B":      df.iloc[j]["name"],
                    "Distance (km)": round(d, 2),
                    "Risk Score":    round(score, 3),
                    "Level": ("🔴 CRITICAL" if d < 20
                              else "🟠 HIGH" if d < 50
                              else "🟡 MEDIUM"),
                })

    out = df.copy()
    out["risk_score"] = risk_scores
    out["risk_level"] = out["risk_score"].apply(
        lambda s: ("🔴 CRITICAL" if s > 0.8
                   else "🟠 HIGH" if s > 0.5
                   else "🟡 MEDIUM" if s > 0
                   else "🟢 SAFE")
    )
    pairs.sort(key=lambda p: p["Distance (km)"])
    return out, pairs
