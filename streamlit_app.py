from __future__ import annotations

import io
import math
import os
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gpxpy
import imageio
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


APP_TITLE = "Track Forge"
APP_SUBTITLE = "GPX, FIT, and Garmin Connect to social-ready visuals"
DEFAULT_OUTPUT_SIZE = (1080, 1350)
DEFAULT_TRACK_COLOR = "#FC4C02"
DEFAULT_BACKGROUND = (0, 0, 0, 0)
DEFAULT_FPS = 30
DEFAULT_DURATION_SECONDS = 8
GARMIN_TOKEN_DIR = Path("data") / "garminconnect"

STAT_DEFINITIONS = [
    ("avg_speed", "Avg speed", "kph"),
    ("elevation_gain", "Elevation gain", "m"),
    ("moving_time", "Moving time", ""),
    ("elapsed_time", "Elapsed time", ""),
    ("avg_heart_rate", "Avg heart rate", "bpm"),
    ("highest_point", "Highest point", "m"),
    ("max_heart_rate", "Max heart rate", "bpm"),
    ("avg_watts", "Avg watts", "W"),
    ("calories", "kcal", "kcal"),
]


@dataclass
class ActivityBundle:
    source: str
    title: str
    subtitle: str
    dataframe: pd.DataFrame
    stats: dict[str, Any]


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


def convert_utc_to_string(utc_seconds: float) -> str:
    return datetime.fromtimestamp(utc_seconds, tz=timezone.utc).strftime("%d.%m.%Y %H:%M:%S UTC")


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(float(seconds)))
    hours, remainder = divmod(max(seconds, 0), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def format_value(value: Any, unit: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    if unit == "":
        return format_duration(value)
    if unit in {"m", "W", "bpm", "kcal"}:
        return f"{int(round(float(value)))} {unit}".strip()
    return f"{float(value):.1f} {unit}".strip()


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


def compute_stats(frame: pd.DataFrame, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = fallback or {}
    stats: dict[str, Any] = {}
    elapsed_time = None
    moving_time = None

    if "time" in frame.columns and frame["time"].notna().any():
        start = pd.to_datetime(frame["time"].iloc[0], utc=True, errors="coerce")
        end = pd.to_datetime(frame["time"].iloc[-1], utc=True, errors="coerce")
        if pd.notna(start) and pd.notna(end):
            elapsed_time = float((end - start).total_seconds())

    if elapsed_time is None and "time_s" in frame.columns and frame["time_s"].notna().any():
        elapsed_time = float(frame["time_s"].dropna().iloc[-1] - frame["time_s"].dropna().iloc[0])

    if elapsed_time is None and len(frame) > 1:
        elapsed_time = float(len(frame) - 1)

    moving_time = (
        maybe_seconds(fallback.get("moving_time"))
        or maybe_seconds(fallback.get("duration"))
        or maybe_seconds(fallback.get("elapsed_time"))
        or elapsed_time
    )

    distance_m = float(frame["distance_m"].dropna().iloc[-1]) if frame["distance_m"].notna().any() else float(fallback.get("distance_m", 0.0) or 0.0)
    altitude = frame["altitude_m"].dropna() if "altitude_m" in frame.columns else pd.Series(dtype=float)
    heartrate = frame["heartrate"].dropna() if "heartrate" in frame.columns else pd.Series(dtype=float)
    watts = frame["watts"].dropna() if "watts" in frame.columns else pd.Series(dtype=float)

    elevation_gain = 0.0
    if len(altitude) > 1:
        diffs = altitude.diff().fillna(0.0)
        elevation_gain = float(diffs[diffs > 0].sum())

    avg_speed_kph = None
    if moving_time and moving_time > 0:
        avg_speed_kph = (distance_m / 1000.0) / (moving_time / 3600.0)

    stats["avg_speed"] = avg_speed_kph
    stats["elevation_gain"] = fallback.get("elevation_gain", elevation_gain)
    stats["moving_time"] = moving_time
    stats["elapsed_time"] = elapsed_time
    stats["avg_heart_rate"] = float(heartrate.mean()) if len(heartrate) else fallback.get("avg_heart_rate")
    stats["highest_point"] = float(altitude.max()) if len(altitude) else fallback.get("highest_point")
    stats["max_heart_rate"] = float(heartrate.max()) if len(heartrate) else fallback.get("max_heart_rate")
    stats["avg_watts"] = float(watts.mean()) if len(watts) else fallback.get("avg_watts")
    stats["calories"] = fallback.get("calories")
    stats["distance_m"] = distance_m
    return stats


def maybe_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    return None


def build_stat_table(stats: dict[str, Any], enabled_stats: list[dict[str, Any]]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for row in sorted(enabled_stats, key=lambda item: (item["order"], item["label"])):
        if not row["enabled"]:
            continue
        rows.append((row["label"], format_value(stats.get(row["key"]), row["unit"])))
    return rows


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


def project_track(points: pd.DataFrame, size: tuple[int, int], padding: int = 70) -> np.ndarray:
    width, height = size
    min_lon, max_lon = float(points["lon"].min()), float(points["lon"].max())
    min_lat, max_lat = float(points["lat"].min()), float(points["lat"].max())
    span_lon = max(max_lon - min_lon, 1e-9)
    span_lat = max(max_lat - min_lat, 1e-9)
    usable_width = width - 2 * padding
    usable_height = height - 2 * padding
    scale = min(usable_width / span_lon, usable_height / span_lat)
    xs = padding + (points["lon"].to_numpy() - min_lon) * scale
    ys = height - padding - (points["lat"].to_numpy() - min_lat) * scale
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


def draw_text_box(image: Image.Image, lines: list[tuple[str, str]], color: str, position: tuple[int, int]) -> None:
    draw = ImageDraw.Draw(image)
    x, y = position
    title_font = get_font(28)
    value_font = get_font(24)
    box_width = 320
    line_height = 34
    padding = 18
    height = padding * 2 + len(lines) * line_height + 12
    draw.rounded_rectangle((x, y, x + box_width, y + height), radius=22, fill=(11, 13, 16, 160), outline=(255, 255, 255, 35), width=1)
    draw.text((x + padding, y + 10), "Session stats", fill=(250, 251, 252, 235), font=title_font)
    offset_y = y + 52
    accent = to_rgba(color)
    for label, value in lines:
        draw.text((x + padding, offset_y), label, fill=(226, 230, 235, 255), font=value_font)
        draw.text((x + box_width - 18 - draw.textlength(value, font=value_font), offset_y), value, fill=accent, font=value_font)
        offset_y += line_height


def draw_footer(image: Image.Image, title: str, subtitle: str) -> None:
    draw = ImageDraw.Draw(image)
    title_font = get_font(48)
    subtitle_font = get_font(26)
    draw.text((56, 44), title, fill=(250, 251, 252, 255), font=title_font)
    draw.text((58, 102), subtitle, fill=(255, 255, 255, 180), font=subtitle_font)


def draw_progress(image: Image.Image, points_xy: np.ndarray, progress: float, color: str, width: int = 9) -> None:
    draw = ImageDraw.Draw(image)
    limit = max(2, int(round((len(points_xy) - 1) * progress)))
    track = [tuple(point) for point in points_xy[:limit]]
    if len(track) > 1:
        draw.line(track, fill=to_rgba(color, 255), width=width, joint="curve")
    marker = tuple(points_xy[min(limit - 1, len(points_xy) - 1)])
    radius = 13
    draw.ellipse((marker[0] - radius, marker[1] - radius, marker[0] + radius, marker[1] + radius), fill=(255, 255, 255, 235))
    draw.ellipse((marker[0] - radius + 4, marker[1] - radius + 4, marker[0] + radius - 4, marker[1] + radius - 4), fill=to_rgba(color, 255))


def render_frame(points: pd.DataFrame, stats_rows: list[tuple[str, str]], config: dict[str, Any], progress: float) -> Image.Image:
    size = config["output_size"]
    background_mode = config["background_mode"]
    if background_mode == "transparent":
        base = Image.new("RGBA", size, DEFAULT_BACKGROUND)
        if config["subtle_grid"]:
            grid = Image.new("RGBA", size, (0, 0, 0, 0))
            grid_draw = ImageDraw.Draw(grid)
            grid_color = (255, 255, 255, 18)
            step = 120
            for x in range(0, size[0], step):
                grid_draw.line((x, 0, x, size[1]), fill=grid_color, width=1)
            for y in range(0, size[1], step):
                grid_draw.line((0, y, size[0], y), fill=grid_color, width=1)
            base = Image.alpha_composite(base, grid)
    else:
        base = render_map_background(points, size, background_mode)

    points_xy = resample_polyline(project_track(points, size), target_points=max(400, int(config["fps"] * config["duration_seconds"] * 6)))
    draw_progress(base, points_xy, progress, config["track_color"], width=config["track_width"])
    draw_footer(base, config["title"], config["subtitle"])
    if config["show_stats"] and stats_rows:
        draw_text_box(base, stats_rows, config["track_color"], position=(56, size[1] - 56 - (34 * len(stats_rows) + 58)))
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


def encode_video(frames: list[Image.Image], fps: int, kind: str) -> bytes:
    suffix = ".webm" if kind == "webm" else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = temp_file.name
    try:
        codec = "libvpx-vp9" if kind == "webm" else "libx264"
        pix_fmt = "yuva420p" if kind == "webm" else "yuv420p"
        with imageio.get_writer(temp_path, fps=fps, codec=codec, macro_block_size=None, ffmpeg_log_level="error", output_params=["-pix_fmt", pix_fmt]) as writer:
            for frame in frames:
                writer.append_data(np.array(frame))
        return Path(temp_path).read_bytes()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def render_bundle(points: pd.DataFrame, stats_rows: list[tuple[str, str]], config: dict[str, Any]) -> dict[str, Any]:
    effective_duration = config["duration_seconds"] / max(config["speed_multiplier"], 0.1)
    total_frames = max(12, int(round(config["fps"] * effective_duration)))
    progress_values = np.linspace(0.0, 1.0, total_frames)
    frames = [render_frame(points, stats_rows, config, progress) for progress in progress_values]
    return {
        "still": frames[-1],
        "frames": frames,
        "gif": encode_gif(frames, config["fps"]),
        "webm": encode_video(frames, config["fps"], "webm"),
        "mp4": encode_video(frames, config["fps"], "mp4"),
    }


def parse_gpx_bytes(data: bytes) -> pd.DataFrame:
    gpx = gpxpy.parse(data.decode("utf-8", errors="ignore"))
    rows: list[dict[str, Any]] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                rows.append({"time": point.time, "lat": point.latitude, "lon": point.longitude, "altitude_m": point.elevation})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    return normalize_activity_frame(frame)


def parse_gpx_upload(uploaded_file: Any) -> pd.DataFrame:
    return parse_gpx_bytes(uploaded_file.getvalue())


def parse_fit_bytes(data: bytes) -> pd.DataFrame:
    fit_file = FitFile(io.BytesIO(data))
    fit_file.parse()
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
        return frame
    if "timestamp" in frame.columns:
        frame["time"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return normalize_activity_frame(frame)


def parse_fit_upload(uploaded_file: Any) -> pd.DataFrame:
    return parse_fit_bytes(uploaded_file.getvalue())


def parse_uploaded_activity(uploaded_file: Any) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".gpx":
        return parse_gpx_upload(uploaded_file)
    if suffix == ".fit":
        return parse_fit_upload(uploaded_file)
    st.error("Only .gpx and .fit files are supported.")
    return pd.DataFrame()


def unpack_downloaded_activity(data: bytes) -> tuple[str, bytes]:
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                suffix = Path(name).suffix.lower()
                if suffix in {".gpx", ".fit", ".tcx"}:
                    return suffix, archive.read(name)
    text = data.lstrip()
    if text.startswith(b"<?xml") or b"<gpx" in text[:2000].lower():
        return ".gpx", data
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
        "avg_watts": activity.get("averagePower") or activity.get("averageWatts"),
        "highest_point": activity.get("maxElevation") or activity.get("highestPoint"),
        "calories": activity.get("calories") or activity.get("totalKilocalories"),
    }


def garmin_download_frame(client: Garmin, activity_id: str) -> pd.DataFrame:
    downloaded = client.download_activity(activity_id, dl_fmt=Garmin.ActivityDownloadFormat.GPX)
    suffix, payload = unpack_downloaded_activity(downloaded)
    if suffix == ".fit":
        return parse_fit_bytes(payload)
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
        frame = garmin_download_frame(client, selected_id)
    except Exception as exc:
        st.error(f"Garmin download failed for activity {selected_id}: {exc}")
        return None

    stats = compute_stats(frame, fallback=garmin_stats(activity))
    return ActivityBundle(
        source="garmin",
        title=activity.get("activityName") or activity.get("name") or "Garmin activity",
        subtitle=f"{activity.get('activityType', {}).get('typeKey') if isinstance(activity.get('activityType'), dict) else activity.get('activityType', 'Garmin')} · {str(activity.get('startTimeLocal') or activity.get('startTimeGMT') or '')[:19]}",
        dataframe=frame,
        stats=stats,
    )


def uploaded_bundle(uploaded_file: Any) -> ActivityBundle | None:
    if uploaded_file is None:
        return None
    frame = parse_uploaded_activity(uploaded_file)
    if frame.empty or not {"lat", "lon"}.issubset(frame.columns):
        st.error("The uploaded file did not contain enough GPS data to render a route.")
        return None
    stats = compute_stats(frame)
    title = Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ").title()
    subtitle = f"Uploaded {Path(uploaded_file.name).suffix.upper().lstrip('.')}"
    return ActivityBundle(source="upload", title=title, subtitle=subtitle, dataframe=frame, stats=stats)


def stat_editor_defaults(stats: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for order, (key, label, unit) in enumerate(STAT_DEFINITIONS, start=1):
        rows.append({"key": key, "label": label, "unit": unit, "enabled": key in {"avg_speed", "elevation_gain", "moving_time"}, "order": order, "value": stats.get(key)})
    return pd.DataFrame(rows)


def sidebar_controls(bundle: ActivityBundle) -> dict[str, Any]:
    st.sidebar.markdown("### Render settings")
    output_size_name = st.sidebar.selectbox("Canvas size", ["Portrait 1080x1350", "Square 1080x1080", "Story 1080x1920"], index=0)
    size_lookup = {"Portrait 1080x1350": (1080, 1350), "Square 1080x1080": (1080, 1080), "Story 1080x1920": (1080, 1920)}
    background_mode = st.sidebar.selectbox("Background", ["transparent", "bw"], format_func=lambda value: "Transparent" if value == "transparent" else "Black & white map", index=0)
    color = st.sidebar.color_picker("Track color", value=DEFAULT_TRACK_COLOR)
    track_width = st.sidebar.slider("Track width", min_value=3, max_value=18, value=9, step=1)
    fps = st.sidebar.slider("Animation FPS", min_value=12, max_value=60, value=DEFAULT_FPS, step=3)
    duration_seconds = st.sidebar.slider("Total duration (seconds)", min_value=2, max_value=30, value=DEFAULT_DURATION_SECONDS, step=1)
    speed_multiplier = st.sidebar.slider("Animation speed", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    subtle_grid = st.sidebar.toggle("Add subtle grid to transparent exports", value=True)
    show_stats = st.sidebar.toggle("Show stats", value=True)
    stats_df = st.sidebar.data_editor(
        stat_editor_defaults(bundle.stats),
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        key="stats_editor",
        column_config={
            "key": st.column_config.TextColumn("Key", disabled=True),
            "label": st.column_config.TextColumn("Label"),
            "unit": st.column_config.TextColumn("Unit", disabled=True),
            "enabled": st.column_config.CheckboxColumn("On"),
            "order": st.column_config.NumberColumn("Order", min_value=1, step=1),
            "value": st.column_config.TextColumn("Preview value", disabled=True),
        },
    )
    stats_rows = [row.to_dict() for _, row in stats_df.iterrows()]
    title = st.sidebar.text_input("Title", value=bundle.title)
    subtitle = st.sidebar.text_input("Subtitle", value=bundle.subtitle)
    return {
        "output_size": size_lookup[output_size_name],
        "background_mode": background_mode,
        "track_color": color,
        "track_width": track_width,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "speed_multiplier": speed_multiplier,
        "subtle_grid": subtle_grid,
        "show_stats": show_stats,
        "stats_rows": stats_rows,
        "title": title,
        "subtitle": subtitle,
    }


def bundle_overview(bundle: ActivityBundle) -> None:
    st.subheader(bundle.title)
    st.caption(bundle.subtitle)
    cols = st.columns(4)
    stat_cards = [
        ("Distance", format_value(bundle.stats.get("distance_m", 0.0) / 1000.0 if bundle.stats.get("distance_m") is not None else None, "km")),
        ("Avg speed", format_value(bundle.stats.get("avg_speed"), "kph")),
        ("Moving time", format_value(bundle.stats.get("moving_time"), "")),
        ("Elevation gain", format_value(bundle.stats.get("elevation_gain"), "m")),
    ]
    for column, (label, value) in zip(cols, stat_cards, strict=False):
        column.metric(label, value)
    with st.expander("All computed stats", expanded=False):
        st.dataframe(pd.DataFrame([{ "metric": key, "value": value } for key, value in bundle.stats.items()]), use_container_width=True, hide_index=True)


def export_panel(bundle: ActivityBundle, rendered: dict[str, Any]) -> None:
    preview, downloads = st.columns([1.35, 1])
    with preview:
        st.markdown("### Preview")
        st.image(rendered["still"], use_container_width=True)
        st.caption("The preview uses the current canvas, track color, and stat settings.")
    with downloads:
        st.markdown("### Downloads")
        still_buffer = io.BytesIO()
        rendered["still"].save(still_buffer, format="PNG")
        st.download_button("Download PNG", data=still_buffer.getvalue(), file_name=f"{bundle.title}.png", mime="image/png", use_container_width=True)
        st.download_button("Download GIF", data=rendered["gif"], file_name=f"{bundle.title}.gif", mime="image/gif", use_container_width=True)
        st.download_button("Download WebM", data=rendered["webm"], file_name=f"{bundle.title}.webm", mime="video/webm", use_container_width=True)
        st.download_button("Download MP4", data=rendered["mp4"], file_name=f"{bundle.title}.mp4", mime="video/mp4", use_container_width=True)
        st.info("GIF and WebM preserve transparency best. MP4 is included for compatibility and may be flattened by some players.")


def main() -> None:
    set_page()
    st.markdown(f"<div class='hero'><h1>{APP_TITLE}</h1><p>{APP_SUBTITLE}</p></div>", unsafe_allow_html=True)

    st.sidebar.markdown("### Source")
    source_mode = st.sidebar.radio("Input", ["Upload GPX/FIT", "Garmin Connect"], horizontal=False)
    bundle: ActivityBundle | None = None

    if source_mode == "Upload GPX/FIT":
        upload = st.sidebar.file_uploader("Upload GPX or FIT", type=["gpx", "fit"], accept_multiple_files=False)
        bundle = uploaded_bundle(upload)
    else:
        st.sidebar.caption("Use personal Garmin Connect credentials for your own activities.")
        client = garmin_login()
        if client is not None:
            bundle = garmin_activity_bundle(client, limit=100)
        else:
            st.sidebar.info("Enter Garmin credentials and click Login to fetch the latest 100 activities.")

    if bundle is None:
        st.info("Upload a GPX/FIT file or log in to Garmin Connect to start.")
        return

    if bundle.dataframe.empty or not {"lat", "lon"}.issubset(bundle.dataframe.columns):
        st.error("The activity data does not contain enough GPS coordinates to render a route.")
        return

    bundle_overview(bundle)
    config = sidebar_controls(bundle)

    with st.spinner("Rendering assets..."):
        rendered = render_bundle(bundle.dataframe, build_stat_table(bundle.stats, config["stats_rows"]), config)

    export_panel(bundle, rendered)
    st.markdown("### Track data")
    st.dataframe(bundle.dataframe.head(200), use_container_width=True)


if __name__ == "__main__":
    main()