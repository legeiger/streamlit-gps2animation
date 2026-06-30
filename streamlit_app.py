from __future__ import annotations

import io
import math
import os
import random
import re
import tempfile
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gpxpy
import imageio
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from fitparse import FitFile
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)
from PIL import Image, ImageColor, ImageDraw, ImageFont


APP_TITLE = "GPS 2 Animation"
APP_SUBTITLE = "GPX, FIT, TCX, and Garmin Connect to social-ready visuals"
DEFAULT_OUTPUT_SIZE = (1024, 1024)
DEFAULT_TRACK_COLOR = "#fc5200"
DEFAULT_BACKGROUND = (0, 0, 0, 0)
DEFAULT_FPS = 30
DEFAULT_DURATION_SECONDS = 5
DEFAULT_SPEED_MULTIPLIER = 3
DEFAULT_STAT_FONT_SIZE = 30
GARMIN_TOKEN_DIR = Path("data") / "garminconnect"

CANVAS_OPTIONS = {
    "Square 640x640": (640, 640),
    "Square 1024x1024": (1024, 1024),
    "Portrait 1080x1350": (1080, 1350),
    "Wide 1920x1080": (1920, 1080),
    "Story 1080x1920": (1080, 1920),
}

# (Key, Default Label, Default Unit, Default Decimals)
STAT_DEFINITIONS = [
    ("distance_m", "Distance", "km", 1),
    ("moving_time", "Time", "", 0),
    ("avg_speed", "Speed/Pace", "kph", 1), # Dynamically overridden
    ("elevation_gain", "Elev Gain", "m", 0),
    ("avg_heart_rate", "Avg HR", "bpm", 0),
    ("avg_cadence", "Avg Cadence", "rpm", 0),
    ("elapsed_time", "Elapsed time", "", 0),
    ("highest_point", "Highest point", "m", 0),
    ("max_heart_rate", "Max HR", "bpm", 0),
    ("avg_watts", "Avg Power", "W", 0),
    ("calories", "Calories", "kcal", 0),
]


@dataclass
class ActivityBundle:
    source: str
    title: str
    subtitle: str
    date_str: str
    activity_type: str
    dataframe: pd.DataFrame
    fallback_stats: dict[str, Any]
    stats: dict[str, Any] = None


def set_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(252, 76, 2, 0.14), transparent 30%),
                linear-gradient(180deg, #0b0d10 0%, #11151a 100%);
            color: #f5f7fa;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }
        .hero {
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            background: rgba(16, 20, 25, 0.75);
            box-shadow: 0 20px 60px rgba(0,0,0,0.30);
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0; font-size: 2.1rem; }
        .hero p { margin: 0.35rem 0 0; color: rgba(245,247,250,0.75); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_secrets_path(*keys: str) -> Any:
    current: Any = st.secrets
    for key in keys:
        try:
            current = current[key]
        except Exception:
            return None
    return current


def format_duration(seconds: float | int | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "-"
    seconds = int(round(float(seconds)))
    hours, remainder = divmod(max(seconds, 0), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def format_value(value: Any, unit: str, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "-"
    val_f = float(value)
    
    if unit == "km":
        return f"{val_f / 1000.0:.{decimals}f} km"
        
    if unit == "kph":
        return f"{val_f:.{decimals}f} kph"
        
    if unit in {"/km", "min/km"}:
        if val_f <= 0:
            return "-"
        pace_seconds = 3600.0 / val_f
        return f"{format_duration(pace_seconds)}/km"
            
    if unit == "":
        return format_duration(value)
        
    if unit in {"m", "W", "bpm", "kcal", "rpm"}:
        return f"{val_f:.{decimals}f} {unit}".strip()
        
    return f"{val_f:.{decimals}f} {unit}".strip()


def to_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    rgb = ImageColor.getrgb(color)
    return rgb[0], rgb[1], rgb[2], alpha


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def cumulative_distance(points: pd.DataFrame) -> pd.Series:
    if len(points) < 2:
        return pd.Series([0.0] * len(points), index=points.index, dtype=float)
    try:
        geometry = gpd.points_from_xy(points["lon"], points["lat"], crs="EPSG:4326")
        gdf = gpd.GeoDataFrame(points[["lat", "lon"]].copy(), geometry=geometry, crs="EPSG:4326")
        projected = gdf.to_crs(gdf.estimate_utm_crs())
        xs = projected.geometry.x.to_numpy(dtype=float)
        ys = projected.geometry.y.to_numpy(dtype=float)
        segment_lengths = np.hypot(np.diff(xs), np.diff(ys))
        distances = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        return pd.Series(distances, index=points.index, dtype=float)
    except Exception:
        distances = [0.0]
        for idx in range(1, len(points)):
            previous = points.iloc[idx - 1]
            current = points.iloc[idx]
            distances.append(
                distances[-1]
                + haversine_m(
                    float(previous["lat"]),
                    float(previous["lon"]),
                    float(current["lat"]),
                    float(current["lon"]),
                )
            )
        return pd.Series(distances, index=points.index, dtype=float)


def normalize_activity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if {"lat", "lon"}.issubset(frame.columns):
        frame = frame.dropna(subset=["lat", "lon"])
    if "time" in frame.columns:
        frame = frame.sort_values("time").reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)
    if "distance_m" not in frame.columns:
        frame["distance_m"] = cumulative_distance(frame)
    else:
        frame["distance_m"] = pd.to_numeric(frame["distance_m"], errors="coerce")
        if frame["distance_m"].isna().all():
            frame["distance_m"] = cumulative_distance(frame)
    if "altitude_m" in frame.columns:
        frame["altitude_m"] = pd.to_numeric(frame["altitude_m"], errors="coerce")
    for column in ["heartrate", "cadence", "watts"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def compute_stats(frame: pd.DataFrame, fallback: dict[str, Any] | None = None, stopped_threshold_ms: float = 0.5) -> dict[str, Any]:
    fallback = fallback or {}
    stats: dict[str, Any] = {}
    
    elapsed_time = fallback.get("elapsed_time")
    moving_time = fallback.get("moving_time")

    if "time" in frame.columns and frame["time"].notna().any() and "distance_m" in frame.columns:
        time_diff = frame["time"].diff().dt.total_seconds().fillna(0)
        dist_diff = frame["distance_m"].diff().fillna(0)
        
        speeds = np.divide(dist_diff, time_diff, out=np.zeros_like(dist_diff), where=time_diff!=0)
        frame["speed_m_s"] = speeds

        if elapsed_time is None:
            start = pd.to_datetime(frame["time"].iloc[0], utc=True, errors="coerce")
            end = pd.to_datetime(frame["time"].iloc[-1], utc=True, errors="coerce")
            if pd.notna(start) and pd.notna(end):
                elapsed_time = float((end - start).total_seconds())
                
        # Re-calc moving time dynamically based on the threshold parameter
        moving_mask = speeds > stopped_threshold_ms
        moving_time = float(time_diff[moving_mask].sum())

    distance_m = float(frame["distance_m"].dropna().iloc[-1]) if frame["distance_m"].notna().any() else float(fallback.get("distance_m", 0.0) or 0.0)
    altitude = frame["altitude_m"].dropna() if "altitude_m" in frame.columns else pd.Series(dtype=float)
    heartrate = frame["heartrate"].dropna() if "heartrate" in frame.columns else pd.Series(dtype=float)
    cadence = frame["cadence"].dropna() if "cadence" in frame.columns else pd.Series(dtype=float)
    watts = frame["watts"].dropna() if "watts" in frame.columns else pd.Series(dtype=float)

    elevation_gain = 0.0
    if len(altitude) > 1:
        diffs = altitude.diff().fillna(0.0)
        elevation_gain = float(diffs[diffs > 0].sum())

    avg_speed_kph = None
    if moving_time and moving_time > 0:
        avg_speed_kph = (distance_m / 1000.0) / (moving_time / 3600.0)

    stats["avg_speed"] = avg_speed_kph if avg_speed_kph else fallback.get("avg_speed")
    stats["elevation_gain"] = elevation_gain if elevation_gain > 0 else fallback.get("elevation_gain")
    stats["moving_time"] = moving_time
    stats["elapsed_time"] = elapsed_time
    
    stats["avg_heart_rate"] = fallback.get("avg_heart_rate") or (float(heartrate.mean()) if len(heartrate) else None)
    stats["max_heart_rate"] = fallback.get("max_heart_rate") or (float(heartrate.max()) if len(heartrate) else None)
    stats["avg_cadence"] = fallback.get("avg_cadence") or (float(cadence.mean()) if len(cadence) else None)
    stats["avg_watts"] = fallback.get("avg_watts") or (float(watts.mean()) if len(watts) else None)
    stats["highest_point"] = fallback.get("highest_point") or (float(altitude.max()) if len(altitude) else None)
    
    stats["calories"] = fallback.get("calories")
    stats["distance_m"] = distance_m
    
    return stats


def trim_track_points(points: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    track_points = points.copy()
    if track_points.empty or not config["privacy_filter_enabled"]:
        return track_points

    start_trim_m = config["privacy_start_trim_m"]
    end_trim_m = config["privacy_end_trim_m"]

    if start_trim_m == 0 and end_trim_m == 0:
        return track_points
        
    max_dist = float(track_points["distance_m"].iloc[-1])
    
    trimmed = track_points[
        (track_points["distance_m"] >= start_trim_m) & 
        (track_points["distance_m"] <= (max_dist - end_trim_m))
    ]
    
    if trimmed.empty:
        return track_points.iloc[:1].reset_index(drop=True)
        
    return trimmed.reset_index(drop=True)


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def mercator_project(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    sin_lat = math.sin(math.radians(lat))
    scale = 256 * (2**zoom)
    x = (lon + 180.0) / 360.0 * scale
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * scale
    return x, y


def choose_zoom(points: pd.DataFrame, width: int, height: int, padding: float = 0.12) -> int:
    lat_min = float(points["lat"].min())
    lat_max = float(points["lat"].max())
    lon_min = float(points["lon"].min())
    lon_max = float(points["lon"].max())
    for zoom in range(16, 1, -1):
        x1, y1 = mercator_project(lat_min, lon_min, zoom)
        x2, y2 = mercator_project(lat_max, lon_max, zoom)
        if abs(x2 - x1) * (1 + padding) <= width and abs(y2 - y1) * (1 + padding) <= height:
            return zoom
    return 2


def tile_url(style: str, z: int, x: int, y: int) -> str:
    if style == "bw":
        return f"https://cartodb-basemaps-a.global.ssl.fastly.net/light_nolabels/{z}/{x}/{y}.png"
    return f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def fetch_tile_image(url: str, timeout: int = 15) -> Image.Image:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return Image.open(io.BytesIO(data)).convert("RGBA")


@st.cache_data(show_spinner=False)
def cached_tile(url: str) -> Image.Image:
    return fetch_tile_image(url)


def render_map_background(points: pd.DataFrame, size: tuple[int, int], style: str) -> Image.Image:
    width, height = size
    zoom = choose_zoom(points, width, height)
    lat_min = float(points["lat"].min())
    lat_max = float(points["lat"].max())
    lon_min = float(points["lon"].min())
    lon_max = float(points["lon"].max())

    min_x, min_y = mercator_project(lat_min, lon_min, zoom)
    max_x, max_y = mercator_project(lat_max, lon_max, zoom)
    pad_x = max(48, int((max_x - min_x) * 0.12))
    pad_y = max(48, int((max_y - min_y) * 0.12))
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    tile_size = 256
    tile_min_x = int(math.floor(min_x / tile_size))
    tile_max_x = int(math.floor(max_x / tile_size))
    tile_min_y = int(math.floor(min_y / tile_size))
    tile_max_y = int(math.floor(max_y / tile_size))

    canvas = Image.new("RGBA", ((tile_max_x - tile_min_x + 1) * tile_size, (tile_max_y - tile_min_y + 1) * tile_size), (18, 21, 26, 255))
    for tile_x in range(tile_min_x, tile_max_x + 1):
        for tile_y in range(tile_min_y, tile_max_y + 1):
            try:
                tile = cached_tile(tile_url(style, zoom, tile_x, tile_y))
                canvas.paste(tile, ((tile_x - tile_min_x) * tile_size, (tile_y - tile_min_y) * tile_size))
            except Exception:
                continue

    crop_left = int(min_x - tile_min_x * tile_size)
    crop_top = int(min_y - tile_min_y * tile_size)
    crop = canvas.crop((crop_left, crop_top, crop_left + width, crop_top + height))
    if crop.size != size:
        crop = crop.resize(size, Image.Resampling.LANCZOS)
    return crop


def project_track(points: pd.DataFrame, size: tuple[int, int], padding: tuple[int, int, int, int] = (44, 44, 44, 44)) -> np.ndarray:
    width, height = size
    left_pad, top_pad, right_pad, bottom_pad = padding
    min_lon, max_lon = float(points["lon"].min()), float(points["lon"].max())
    min_lat, max_lat = float(points["lat"].min()), float(points["lat"].max())
    span_lon = max(max_lon - min_lon, 1e-9)
    span_lat = max(max_lat - min_lat, 1e-9)
    usable_width = max(1, width - left_pad - right_pad)
    usable_height = max(1, height - top_pad - bottom_pad)
    scale = min(usable_width / span_lon, usable_height / span_lat)
    track_width = span_lon * scale
    track_height = span_lat * scale
    x_offset = left_pad + (usable_width - track_width) / 2.0
    y_offset = top_pad + (usable_height - track_height) / 2.0
    xs = x_offset + (points["lon"].to_numpy() - min_lon) * scale
    ys = height - y_offset - (points["lat"].to_numpy() - min_lat) * scale
    return np.column_stack([xs, ys])


def resample_polyline(points_xy: np.ndarray, target_points: int = 800) -> np.ndarray:
    if len(points_xy) <= 2 or target_points <= len(points_xy):
        return points_xy
    deltas = np.sqrt(np.sum(np.diff(points_xy, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    if cumulative[-1] == 0:
        return points_xy
    sample = np.linspace(0, cumulative[-1], target_points)
    x = np.interp(sample, cumulative, points_xy[:, 0])
    y = np.interp(sample, cumulative, points_xy[:, 1])
    return np.column_stack([x, y])


def draw_progress(image: Image.Image, points_xy: np.ndarray, progress: float, color: str, width: int = 9) -> None:
    draw = ImageDraw.Draw(image)
    limit = max(2, int(round((len(points_xy) - 1) * progress)))
    track = [tuple(point) for point in points_xy[:limit]]
    if len(track) > 1:
        draw.line(track, fill=to_rgba(color, 255), width=width, joint="curve")


def render_frame(points: pd.DataFrame, config: dict[str, Any], progress: float) -> Image.Image:
    size = config["output_size"]
    background_mode = config["background_mode"]
    
    if background_mode == "transparent":
        base = Image.new("RGBA", size, DEFAULT_BACKGROUND)
    else:
        base = render_map_background(points, size, background_mode)
        
    points_xy = resample_polyline(
        project_track(points, size),
        target_points=len(points),
    )
    draw_progress(base, points_xy, progress, config["track_color"], width=config["track_width"])
    return base


def encode_gif(frames: list[Image.Image], fps: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp_file:
        temp_path = temp_file.name
    try:
        imageio.mimsave(temp_path, [np.array(frame) for frame in frames], duration=1 / max(fps, 1), loop=0)
        return Path(temp_path).read_bytes()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def encode_webm(frames: list[Image.Image], fps: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
        temp_path = temp_file.name
    try:
        with imageio.get_writer(temp_path, fps=fps, codec="libvpx-vp9", macro_block_size=None, ffmpeg_log_level="error", output_params=["-pix_fmt", "yuva420p"]) as writer:
            for frame in frames:
                writer.append_data(np.array(frame))
        return Path(temp_path).read_bytes()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def render_bundle(points: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    effective_duration = config["duration_seconds"] / max(config["speed_multiplier"], 0.1)
    total_frames = max(12, int(round(config["fps"] * effective_duration)))
    progress_values = np.linspace(0.0, 1.0, total_frames)
    track_points = trim_track_points(points, config)
    
    frames = [render_frame(track_points, config, progress) for progress in progress_values]
    
    pause_frames = int(round(config["fps"] * config.get("end_pause_seconds", 0)))
    if pause_frames > 0 and frames:
        frames.extend([frames[-1]] * pause_frames)

    return {
        "still": frames[-1],
        "frames": frames,
        "gif": encode_gif(frames, config["fps"]),
        "webm": encode_webm(frames, config["fps"]),
    }


def render_stats_png(bundle: ActivityBundle, config: dict[str, Any]) -> bytes:
    """Renders the stats UI dynamically based on the enabled stats and custom decimals."""
    width, height = 800, 360
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    font_size = config["stats_font_size"]
    drop_shadow = config["stats_png_drop_shadow"]
    
    enabled_stats = [row for row in config["stats_rows"] if row.get("enabled")]
    
    if not enabled_stats:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
        
    num_stats = len(enabled_stats)
    cols = 2
    rows = math.ceil(num_stats / cols)
    
    value_font = get_font(font_size + 12)
    label_font = get_font(max(12, font_size - 4))
    
    pad_x, pad_y = 20, 20
    box_w = (width - pad_x * (cols + 1)) // cols
    box_h = (height - pad_y * (rows + 1)) // rows
    
    for i, stat_row in enumerate(enabled_stats):
        col = i % cols
        r = i // cols
        
        x0 = pad_x + col * (box_w + pad_x)
        
        if i == num_stats - 1 and num_stats % 2 != 0:
            x0 = (width - box_w) / 2
            
        y0 = pad_y + r * (box_h + pad_y)
        
        label = stat_row["label"]
        raw_val = bundle.stats.get(stat_row["key"])
        
        # Apply the user's custom decimal choice when rendering!
        value = format_value(raw_val, stat_row["unit"], stat_row.get("decimals", 0))
            
        _, _, lx, ly = draw.textbbox((0, 0), label, font=label_font)
        label_x = x0 + (box_w - lx) / 2
        label_y = y0 + box_h / 2 - ly - 5
        
        _, _, vx, vy = draw.textbbox((0, 0), value, font=value_font)
        val_x = x0 + (box_w - vx) / 2
        val_y = y0 + box_h / 2 + 5
        
        if drop_shadow:
            shadow_offset = max(1, font_size // 15)
            shadow_color = (0, 0, 0, 180)
            draw.text((label_x + shadow_offset, label_y + shadow_offset), label, font=label_font, fill=shadow_color)
            draw.text((val_x + shadow_offset, val_y + shadow_offset), value, font=value_font, fill=shadow_color)
            
        draw.text((label_x, label_y), label, font=label_font, fill=(255, 255, 255, 255))
        draw.text((val_x, val_y), value, font=value_font, fill=(255, 255, 255, 255))
        
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def parse_gpx_bytes(data: bytes) -> tuple[pd.DataFrame, str]:
    gpx = gpxpy.parse(data.decode("utf-8", errors="ignore"))
    rows: list[dict[str, Any]] = []
    
    activity_type = "unknown"
    if gpx.tracks and gpx.tracks[0].type:
        activity_type = gpx.tracks[0].type
    
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                row = {
                    "time": point.time, 
                    "lat": point.latitude, 
                    "lon": point.longitude, 
                    "altitude_m": point.elevation
                }
                
                # Extract extensions (e.g. Heart Rate, Cadence)
                for ext in point.extensions:
                    for child in ext:
                        tag = child.tag.lower()
                        if child.text is not None:
                            try:
                                if 'hr' in tag or 'heartrate' in tag:
                                    row['heartrate'] = float(child.text)
                                elif 'cad' in tag or 'cadence' in tag:
                                    row['cadence'] = float(child.text)
                                elif 'power' in tag or 'watts' in tag:
                                    row['watts'] = float(child.text)
                            except ValueError:
                                pass
                                
                rows.append(row)
                
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, activity_type
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    return normalize_activity_frame(frame), activity_type


def parse_tcx_bytes(data: bytes) -> tuple[pd.DataFrame, str]:
    activity_type = "unknown"
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        st.error(f"Failed to parse TCX: {e}")
        return pd.DataFrame(), activity_type
        
    rows = []
    ns = {
        'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
        'ext': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'
    }
    
    activity_node = root.find('.//tcx:Activity', ns)
    if activity_node is not None:
        activity_type = activity_node.attrib.get('Sport', 'unknown')
    
    for trackpoint in root.findall('.//tcx:Trackpoint', ns):
        row = {}
        
        time_el = trackpoint.find('tcx:Time', ns)
        if time_el is not None: 
            row['time'] = time_el.text
            
        pos = trackpoint.find('tcx:Position', ns)
        if pos is not None:
            lat = pos.find('tcx:LatitudeDegrees', ns)
            lon = pos.find('tcx:LongitudeDegrees', ns)
            if lat is not None: row['lat'] = float(lat.text)
            if lon is not None: row['lon'] = float(lon.text)
            
        alt = trackpoint.find('tcx:AltitudeMeters', ns)
        if alt is not None: row['altitude_m'] = float(alt.text)
        
        hr = trackpoint.find('tcx:HeartRateBpm/tcx:Value', ns)
        if hr is not None: row['heartrate'] = float(hr.text)
        
        cad = trackpoint.find('tcx:Cadence', ns)
        if cad is not None: row['cadence'] = float(cad.text)
        
        watts = trackpoint.find('.//ext:Watts', ns)
        if watts is not None: row['watts'] = float(watts.text)
        
        if 'lat' in row and 'lon' in row:
            rows.append(row)
            
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, activity_type
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    return normalize_activity_frame(frame), activity_type


def parse_fit_bytes(data: bytes) -> tuple[pd.DataFrame, str]:
    fit_file = FitFile(io.BytesIO(data))
    fit_file.parse()
    
    activity_type = "unknown"
    for msg in fit_file.get_messages("sport"):
        sport = msg.get_value("sport")
        sub_sport = msg.get_value("sub_sport")
        if sport:
            activity_type = str(sport)
            if sub_sport: activity_type += f"_{sub_sport}"
        break
        
    rows: list[dict[str, Any]] = []
    for message in fit_file.get_messages("record"):
        row: dict[str, Any] = {}
        for field in message:
            row[field.name] = field.value
        if row.get("position_lat") is not None and row.get("position_long") is not None:
            row["lat"] = float(row["position_lat"]) * 180.0 / 2**31
            row["lon"] = float(row["position_long"]) * 180.0 / 2**31
        if row.get("altitude") is not None:
            row["altitude_m"] = float(row["altitude"])
        if row.get("heart_rate") is not None:
            row["heartrate"] = float(row["heart_rate"])
        if row.get("cadence") is not None:
            row["cadence"] = float(row["cadence"])
        if row.get("power") is not None:
            row["watts"] = float(row["power"])
        rows.append(row)
        
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, activity_type
    if "timestamp" in frame.columns:
        frame["time"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return normalize_activity_frame(frame), activity_type


def parse_uploaded_activity(uploaded_file: Any) -> tuple[pd.DataFrame, str]:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".gpx":
        return parse_gpx_bytes(uploaded_file.getvalue())
    if suffix == ".tcx":
        return parse_tcx_bytes(uploaded_file.getvalue())
    if suffix == ".fit":
        return parse_fit_bytes(uploaded_file.getvalue())
    st.error("Only .gpx, .tcx, and .fit files are supported.")
    return pd.DataFrame(), "unknown"


def unpack_downloaded_activity(data: bytes) -> tuple[str, bytes]:
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                suffix = Path(name).suffix.lower()
                if suffix in {".gpx", ".fit", ".tcx"}:
                    return suffix, archive.read(name)
    text = data.lstrip()
    if text.startswith(b"<?xml"):
        if b"<gpx" in text[:2000].lower():
            return ".gpx", data
        if b"<TrainingCenterDatabase" in text[:2000].lower():
            return ".tcx", data
    return ".fit", data


def garmin_stats(activity: dict[str, Any]) -> dict[str, Any]:
    average_speed = activity.get("averageSpeed")
    moving_time = activity.get("duration") or activity.get("durationInSeconds")
    distance_m = activity.get("distance")
    return {
        "distance_m": distance_m,
        "moving_time": moving_time,
        "elapsed_time": moving_time,
        "avg_speed": (float(average_speed) * 3.6) if average_speed is not None else None,
        "elevation_gain": activity.get("elevationGain") or activity.get("totalElevationGain"),
        "avg_heart_rate": activity.get("averageHR") or activity.get("averageHeartRate"),
        "max_heart_rate": activity.get("maxHR") or activity.get("maxHeartRate"),
        "avg_cadence": activity.get("averageRunningCadenceInStepsPerMinute") or activity.get("averageBikingCadenceInRevPerMinute"),
        "avg_watts": activity.get("averagePower") or activity.get("averageWatts"),
        "highest_point": activity.get("maxElevation") or activity.get("highestPoint"),
        "calories": activity.get("calories") or activity.get("totalKilocalories"),
    }


def garmin_download_frame(client: Garmin, activity_id: str) -> tuple[pd.DataFrame, str]:
    downloaded = client.download_activity(activity_id, dl_fmt=Garmin.ActivityDownloadFormat.GPX)
    suffix, payload = unpack_downloaded_activity(downloaded)
    if suffix == ".fit":
        return parse_fit_bytes(payload)
    if suffix == ".tcx":
        return parse_tcx_bytes(payload)
    return parse_gpx_bytes(payload)


def garmin_login() -> Garmin | None:
    if st.session_state.get("garmin_client") is not None:
        return st.session_state["garmin_client"]

    GARMIN_TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    token_store = str(GARMIN_TOKEN_DIR)

    with st.sidebar.form("garmin_login_form", clear_on_submit=False):
        email_default = safe_secrets_path("garmin", "EMAIL") or ""
        password_default = safe_secrets_path("garmin", "PASSWORD") or ""
        email = st.text_input("Garmin email", value=email_default)
        password = st.text_input("Garmin password", value=password_default, type="password")
        submit = st.form_submit_button("Login to Garmin")

    if submit:
        try:
            client = Garmin(email, password, return_on_mfa=True)
            login_result = client.login(token_store)
            if isinstance(login_result, tuple) and len(login_result) == 2:
                mfa_status, state = login_result
            else:
                mfa_status, state = None, None

            if mfa_status == "NEEDS_MFA":
                st.session_state["garmin_pending_client"] = client
                st.session_state["garmin_pending_state"] = state
                st.session_state["garmin_email"] = email
                st.session_state.pop("garmin_client", None)
                st.warning("Garmin requires MFA. Enter the code below to finish login.")
            else:
                st.session_state["garmin_client"] = client
                st.session_state["garmin_email"] = email
                st.session_state.pop("garmin_pending_client", None)
                st.session_state.pop("garmin_pending_state", None)
                st.success("Garmin login ready")
        except GarminConnectTooManyRequestsError:
            st.error("Garmin rate-limited this IP. Wait a few minutes before trying again.")
        except GarminConnectAuthenticationError as exc:
            st.error(f"Garmin authentication failed: {exc}")
        except GarminConnectConnectionError as exc:
            st.error(f"Garmin connection failed: {exc}")
        except Exception as exc:
            st.error(f"Garmin login failed: {exc}")

    pending_client = st.session_state.get("garmin_pending_client")
    pending_state = st.session_state.get("garmin_pending_state")
    if pending_client is not None and pending_state is not None:
        with st.sidebar.form("garmin_mfa_form", clear_on_submit=False):
            mfa_code = st.text_input("MFA code", value="", placeholder="Enter the code from Garmin")
            mfa_submit = st.form_submit_button("Verify MFA")

        if mfa_submit:
            try:
                pending_client.resume_login(pending_state, mfa_code)
                st.session_state["garmin_client"] = pending_client
                st.session_state.pop("garmin_pending_client", None)
                st.session_state.pop("garmin_pending_state", None)
                st.success("Garmin MFA verified")
                return pending_client
            except GarminConnectTooManyRequestsError:
                st.error("Garmin rate-limited this IP while verifying MFA. Wait before retrying.")
            except GarminConnectAuthenticationError as exc:
                st.error(f"Garmin MFA failed: {exc}")
            except GarminConnectConnectionError as exc:
                st.error(f"Garmin connection failed: {exc}")
            except Exception as exc:
                st.error(f"Garmin MFA verification failed: {exc}")

    return st.session_state.get("garmin_client")


def garmin_activity_bundle(client: Garmin, limit: int) -> ActivityBundle | None:
    activities = list(client.get_activities(start=0, limit=limit))
    if not activities:
        st.info("No Garmin activities found.")
        return None

    activity_map = {str(activity.get("activityId")): activity for activity in activities if activity.get("activityId") is not None}
    activity_ids = list(activity_map.keys())

    def option_label(activity_id: str) -> str:
        activity = activity_map[activity_id]
        start = activity.get("startTimeLocal") or activity.get("startTimeGMT") or ""
        start_text = str(start)[:10]
        distance_km = float(activity.get("distance") or 0.0) / 1000.0
        name = activity.get("activityName") or activity.get("name") or "Garmin activity"
        activity_type = (activity.get("activityType") or {}).get("typeKey") if isinstance(activity.get("activityType"), dict) else activity.get("activityType")
        return f"{name} | {start_text} | {distance_km:.1f} km | {activity_type or 'unknown'}"

    selected_id = st.selectbox("Latest Garmin activities", activity_ids, format_func=option_label)
    activity = activity_map[selected_id]

    try:
        frame, parsed_type = garmin_download_frame(client, selected_id)
    except Exception as exc:
        st.error(f"Garmin download failed for activity {selected_id}: {exc}")
        return None
        
    activity_type = activity.get("activityType", {}).get("typeKey", parsed_type)

    date_str = str(activity.get("startTimeLocal") or activity.get("startTimeGMT") or "Unknown")[:10]
    title = activity.get("activityName") or activity.get("name") or "Garmin activity"
    
    return ActivityBundle(
        source="garmin",
        title=title,
        subtitle=f"{activity_type} · {date_str}",
        date_str=date_str,
        activity_type=str(activity_type),
        dataframe=frame,
        fallback_stats=garmin_stats(activity),
    )


def uploaded_bundle(uploaded_file: Any) -> ActivityBundle | None:
    if uploaded_file is None:
        return None
    frame, activity_type = parse_uploaded_activity(uploaded_file)
    if frame.empty or not {"lat", "lon"}.issubset(frame.columns):
        st.error("The uploaded file did not contain enough GPS data to render a route.")
        return None
        
    title = Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ").title()
    subtitle = f"Uploaded {Path(uploaded_file.name).suffix.upper().lstrip('.')}"
    
    date_str = "UnknownDate"
    if "time" in frame.columns and not frame["time"].empty:
        date_obj = pd.to_datetime(frame["time"].iloc[0])
        if pd.notna(date_obj):
            date_str = date_obj.strftime("%Y-%m-%d")

    return ActivityBundle(
        source="upload", 
        title=title, 
        subtitle=subtitle, 
        date_str=date_str,
        activity_type=str(activity_type),
        dataframe=frame, 
        fallback_stats={}
    )


def stat_editor_defaults(stats: dict[str, Any], activity_type: str) -> pd.DataFrame:
    rows = []
    
    is_cycling = any(kw in activity_type.lower() for kw in ["cycl", "bik", "gravel"])
    
    default_keys = ["distance_m", "moving_time", "avg_speed", "elevation_gain"]
    if stats.get("avg_heart_rate"): default_keys.append("avg_heart_rate")
    
    for order, (key, label, unit, decimals) in enumerate(STAT_DEFINITIONS, start=1):
        # Override Speed vs Pace automatically based on activity type
        if key == "avg_speed":
            if is_cycling:
                label = "Speed"
                unit = "kph"
            else:
                label = "Pace"
                unit = "/km"
                decimals = 0 # Min/sec layout doesn't use decimals

        formatted_val = format_value(stats.get(key), unit, decimals)
        rows.append({
            "key": key, 
            "label": label, 
            "unit": unit, 
            "decimals": decimals,
            "enabled": key in default_keys, 
            "order": order, 
            "value": formatted_val
        })
    return pd.DataFrame(rows)


def sidebar_controls(bundle: ActivityBundle) -> dict[str, Any]:
    if "default_start_trim" not in st.session_state:
        st.session_state.default_start_trim = random.randint(100, 750)
        st.session_state.default_end_trim = random.randint(100, 750)

    st.sidebar.markdown("### Render settings")
    output_size_name = st.sidebar.selectbox("Canvas size", list(CANVAS_OPTIONS.keys()), index=1)
    size_lookup = CANVAS_OPTIONS
    background_mode = st.sidebar.selectbox("Background", ["transparent", "bw"], format_func=lambda value: "Transparent" if value == "transparent" else "Black & white map", index=0)
    color = st.sidebar.color_picker("Track color", value=DEFAULT_TRACK_COLOR)
    track_width = st.sidebar.number_input("Track width", min_value=1, max_value=40, value=9, step=1)
    fps = st.sidebar.number_input("Animation FPS", min_value=1, max_value=60, value=DEFAULT_FPS, step=1)
    duration_seconds = st.sidebar.number_input("Total duration (seconds)", min_value=1, max_value=60, value=DEFAULT_DURATION_SECONDS, step=1)
    speed_multiplier = st.sidebar.number_input("Animation speed", min_value=1, max_value=6, value=DEFAULT_SPEED_MULTIPLIER, step=1)
    end_pause_seconds = st.sidebar.number_input("Pause on last frame (seconds)", min_value=0, max_value=10, value=4, step=1)
    
    st.sidebar.markdown("### Data Analytics")
    stopped_threshold_ms = st.sidebar.slider("Stopped speed threshold (m/s)", 0.0, 3.0, 1.1, 0.1, help="Speeds below this threshold will not be counted in moving time.")
    st.sidebar.markdown(f"Speed below {stopped_threshold_ms * 3.6:.1f} kph is considered stopped.")
    # Dynamically re-calculate stats using the user's stopped threshold
    bundle.stats = compute_stats(bundle.dataframe, fallback=bundle.fallback_stats, stopped_threshold_ms=stopped_threshold_ms)
    
    st.sidebar.markdown("### Stat Overlay Settings")
    stats_font_size = st.sidebar.number_input("Stats font size", min_value=8, max_value=64, value=DEFAULT_STAT_FONT_SIZE, step=1)
    stats_png_drop_shadow = st.sidebar.toggle("Stats PNG Drop Shadow", value=True)
    show_stats = st.sidebar.toggle("Show stats in UI", value=True)
    
    st.sidebar.markdown("### Privacy Settings")
    privacy_filter_enabled = st.sidebar.toggle("Privacy filter", value=False)
    privacy_start_trim_m = st.sidebar.slider("Trim from start (meters)", min_value=0, max_value=5000, value=st.session_state.default_start_trim, step=50)
    privacy_end_trim_m = st.sidebar.slider("Trim from end (meters)", min_value=0, max_value=5000, value=st.session_state.default_end_trim, step=50)
    
    # Initialize the data editor layout with auto-detected Pace/Speed labels based on type
    stats_df = st.sidebar.data_editor(
        stat_editor_defaults(bundle.stats, bundle.activity_type),
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        key="stats_editor",
        column_config={
            "key": st.column_config.TextColumn("Key", disabled=True),
            "label": st.column_config.TextColumn("Label"),
            "unit": st.column_config.TextColumn("Unit", disabled=True),
            "decimals": st.column_config.NumberColumn("Decimals", min_value=0, max_value=3, step=1),
            "enabled": st.column_config.CheckboxColumn("On"),
            "order": st.column_config.NumberColumn("Order", min_value=1, step=1),
            "value": st.column_config.TextColumn("Preview value", disabled=True),
        },
    )
    
    stats_rows = [row.to_dict() for _, row in stats_df.iterrows()]
    stats_rows = sorted(stats_rows, key=lambda x: x.get('order', 99))
    
    return {
        "output_size": size_lookup[output_size_name],
        "background_mode": background_mode,
        "track_color": color,
        "track_width": track_width,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "speed_multiplier": speed_multiplier,
        "end_pause_seconds": end_pause_seconds,
        "stats_font_size": stats_font_size,
        "stats_png_drop_shadow": stats_png_drop_shadow,
        "show_stats": show_stats,
        "privacy_filter_enabled": privacy_filter_enabled,
        "privacy_start_trim_m": privacy_start_trim_m,
        "privacy_end_trim_m": privacy_end_trim_m,
        "stats_rows": stats_rows,
    }


def bundle_overview(bundle: ActivityBundle, config: dict[str, Any]) -> None:
    if not config["show_stats"]:
        return
        
    enabled_stats = [row for row in config["stats_rows"] if row.get("enabled")]
    
    st.markdown("<div style='text-align:center; font-size:1.1rem; font-weight:700; letter-spacing:0.02em; margin: 0.25rem 0 0.85rem;'>Stats</div>", unsafe_allow_html=True)
    
    cols = st.columns(2)
    label_font_size = max(10, config["stats_font_size"] - 8)
    card_html = """
    <div style="padding: 1rem 1.1rem; border-radius: 18px; border: 1px solid rgba(255,255,255,0.08); background: rgba(13,17,22,0.72); box-shadow: 0 14px 34px rgba(0,0,0,0.18); text-align:center; display:flex; flex-direction:column; align-items:center; justify-content:center; min-height: 112px; margin-bottom: 1rem;">
      <div style="font-size: {label_font_size}px; line-height: 1.1; letter-spacing: 0.02em; color: rgba(245,247,250,0.72); margin-bottom: 0.45rem;">{label}</div>
      <div style="font-size: {value_font_size}px; line-height: 1.05; font-weight: 700; color: #f5f7fa;">{value}</div>
    </div>
    """
    
    num_stats = len(enabled_stats)
    for i, stat_row in enumerate(enabled_stats):
        col_idx = i % 2
        
        # Center the final item if odd by putting it in a container that spans cols
        if i == num_stats - 1 and num_stats % 2 != 0:
            with st.container():
                st.markdown(
                    card_html.format(
                        label=stat_row["label"], 
                        value=format_value(bundle.stats.get(stat_row["key"]), stat_row["unit"], stat_row.get("decimals", 0)), 
                        label_font_size=label_font_size, 
                        value_font_size=config["stats_font_size"]
                    ),
                    unsafe_allow_html=True,
                )
        else:
            with cols[col_idx]:
                st.markdown(
                    card_html.format(
                        label=stat_row["label"], 
                        value=format_value(bundle.stats.get(stat_row["key"]), stat_row["unit"], stat_row.get("decimals", 0)), 
                        label_font_size=label_font_size, 
                        value_font_size=config["stats_font_size"]
                    ),
                    unsafe_allow_html=True,
                )
                
    with st.expander("All computed stats", expanded=False):
        st.dataframe(pd.DataFrame([{ "metric": key, "value": value } for key, value in bundle.stats.items()]), width="stretch", hide_index=True)

        # Percentiles and Histogram additions
        is_cycling = any(kw in bundle.activity_type.lower() for kw in ["cycl", "bik", "gravel"])
        unit = "kph" if is_cycling else "/km"
        decimals = 1 if is_cycling else 0
        percentiles = [5, 25, 50, 75, 80, 95]
        p_labels = [f"p{p:02d}" for p in percentiles]

        if "speed_m_s" in bundle.dataframe.columns:
            valid_speeds = bundle.dataframe["speed_m_s"][bundle.dataframe["speed_m_s"] > 0]
            if not valid_speeds.empty:
                p_vals = np.percentile(valid_speeds * 3.6, percentiles)
                st.markdown("<div style='text-align:center; font-size:1.1rem; font-weight:700; margin: 1.5rem 0 0.85rem;'>Speed / Pace Percentiles</div>", unsafe_allow_html=True)
                p_cols = st.columns(len(percentiles))
                for col, lbl, val in zip(p_cols, p_labels, p_vals):
                    col.metric(lbl, format_value(val, unit, decimals))

        st.markdown("<div style='text-align:center; font-size:1.1rem; font-weight:700; margin: 1.5rem 0 0.85rem;'>All Metrics Percentiles</div>", unsafe_allow_html=True)
        percentile_data = []
        metrics_map = {
            "speed_m_s": ("Speed/Pace", unit, decimals),
            "heartrate": ("Heart Rate", "bpm", 0),
            "cadence": ("Cadence", "rpm", 0),
            "watts": ("Power", "W", 0),
            "altitude_m": ("Altitude", "m", 0)
        }
        for col_name, (m_label, m_unit, m_dec) in metrics_map.items():
            if col_name in bundle.dataframe.columns:
                valid_data = bundle.dataframe[col_name].dropna()
                if not valid_data.empty and (valid_data > 0).any():
                    if col_name == "speed_m_s":
                        valid_data = valid_data[valid_data > 0] * 3.6
                    p_vals = np.percentile(valid_data, percentiles)
                    row_data = {"Metric": m_label}
                    for lbl, val in zip(p_labels, p_vals):
                        row_data[lbl] = format_value(val, m_unit, m_dec)
                    percentile_data.append(row_data)

        if percentile_data:
            st.dataframe(pd.DataFrame(percentile_data), width="stretch", hide_index=True)

        st.markdown("<div style='text-align:center; font-size:1.1rem; font-weight:700; margin: 1.5rem 0 0.85rem;'>Speed Distribution</div>", unsafe_allow_html=True)
        bins = st.slider("Histogram bins", min_value=5, max_value=50, value=20, step=1)
        if "speed_m_s" in bundle.dataframe.columns:
            valid_speeds_kph = bundle.dataframe["speed_m_s"][bundle.dataframe["speed_m_s"] > 0] * 3.6
            if not valid_speeds_kph.empty:
                counts, bin_edges = np.histogram(valid_speeds_kph, bins=bins)
                shares = counts / counts.sum()
                bin_labels = []
                for i in range(len(counts)):
                    start_str = format_value(bin_edges[i], unit, decimals).replace(unit, '').strip()
                    end_str = format_value(bin_edges[i+1], unit, decimals)
                    bin_labels.append(f"{start_str} - {end_str}")
                
                hist_df = pd.DataFrame({"Relative Share": shares}, index=bin_labels)
                st.bar_chart(hist_df, y="Relative Share")
                
                hist_w, hist_h = 1000, 400
                hist_img = Image.new("RGBA", (hist_w, hist_h), (0,0,0,0))
                draw = ImageDraw.Draw(hist_img)
                max_share = shares.max()
                if max_share > 0:
                    bar_w = hist_w / len(shares)
                    for i, share in enumerate(shares):
                        draw.rectangle([i * bar_w, hist_h - (share / max_share) * hist_h, (i * bar_w) + bar_w * 0.85, hist_h], fill=to_rgba(config["track_color"], 255))
                
                buf = io.BytesIO()
                hist_img.save(buf, format="PNG")
                st.download_button("Download Transparent Histogram PNG", buf.getvalue(), file_name=f"histogram_export.png", mime="image/png", width="stretch")


def format_filename(bundle: ActivityBundle, asset_type: str, suffix: str) -> str:
    safe_title = re.sub(r'[^A-Za-z0-9_-]', '', bundle.title.replace(' ', '_'))
    dist_km = float(bundle.stats.get("distance_m") or 0.0) / 1000.0
    return f"{bundle.date_str}_{safe_title}_{dist_km:.1f}km_{asset_type}.{suffix}"


def export_panel(bundle: ActivityBundle, config: dict[str, Any], rendered: dict[str, Any]) -> None:
    preview, downloads = st.columns([1.35, 1])
    
    # Pre-render the stats image so we can show a preview and download it
    stats_png_bytes = render_stats_png(bundle, config)
    
    with preview:
        st.markdown("### Previews")
        st.image(rendered["still"], width="stretch", caption="Track Asset Preview")
        st.image(stats_png_bytes, width="stretch", caption="Stats Overlay Preview")
        
    with downloads:
        st.markdown("### Exports")
        
        # Track Asset Downloads
        still_buffer = io.BytesIO()
        rendered["still"].save(still_buffer, format="PNG")
        
        st.download_button("Download Track PNG", data=still_buffer.getvalue(), file_name=format_filename(bundle, "track", "png"), mime="image/png", width="stretch")
        st.download_button("Download GIF", data=rendered["gif"], file_name=format_filename(bundle, "track", "gif"), mime="image/gif", width="stretch")
        st.download_button("Download WebM", data=rendered["webm"], file_name=format_filename(bundle, "track", "webm"), mime="video/webm", width="stretch")
        st.info("GIF and WebM preserve transparency best.")
        
        st.divider()
        
        # New Stats Overlay Download
        st.markdown("### Export Stats Overlay")
        st.caption("PNG with selected stats from the sidebar table.")
        st.download_button(
            "Download Stats PNG", 
            data=stats_png_bytes, 
            file_name=format_filename(bundle, "stats", "png"), 
            mime="image/png", 
            width="stretch"
        )
        


def main() -> None:
    set_page()
    st.markdown(f"<div class='hero'><h1>{APP_TITLE}</h1><p>{APP_SUBTITLE}</p></div>", unsafe_allow_html=True)

    source_mode = st.sidebar.radio("Input", ["Upload GPX/FIT/TCX", "Garmin Connect"], horizontal=False)
    bundle: ActivityBundle | None = None

    if source_mode == "Upload GPX/FIT/TCX":
        upload = st.sidebar.file_uploader("Upload GPX, FIT, or TCX", type=["gpx", "fit", "tcx"], accept_multiple_files=False)
        bundle = uploaded_bundle(upload)
    else:
        st.sidebar.caption("Use personal Garmin Connect credentials for your own activities.")
        client = garmin_login()
        if client is not None:
            bundle = garmin_activity_bundle(client, limit=100)
        else:
            st.sidebar.info("Enter Garmin credentials and click Login to fetch the latest 100 activities.")

    if bundle is None:
        st.info("Upload a GPX/FIT/TCX file or log in to Garmin Connect to start.")
        return

    if bundle.dataframe.empty or not {"lat", "lon"}.issubset(bundle.dataframe.columns):
        st.error("The activity data does not contain enough GPS coordinates to render a route.")
        return

    config = sidebar_controls(bundle)
    bundle_overview(bundle, config)

    with st.spinner("Rendering assets..."):
        rendered = render_bundle(bundle.dataframe, config)

    export_panel(bundle, config, rendered)
    
    st.markdown("### Track data")
    st.dataframe(bundle.dataframe.head(100), width="stretch")


if __name__ == "__main__":
    main()