"""
This API implements:
- REST API1: Access to sensor data, facilities, and calendar events
- REST API2: MCDM-based room recommendations using AHP
"""

import sqlite3
import numpy as np
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, field_validator
from contextlib import closing
import uvicorn
from datetime import datetime, timedelta
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# CONFIG
DB_PATH = "sensors.db"

# FASTAPI APP INIT

app = FastAPI(
    title="Room Selection Decision Support System",
    description="""
## Overview
This API provides access to room sensor data, facilities, calendar events, 
and a room recommendation system based on Multi-Criteria Decision Making (MCDM).

## REST API1 - Data Access
- **Sensors**: Temperature, humidity, CO2, sound, light measurements
- **Facilities**: Room equipment (projector, seats, whiteboard, PC)
- **Calendar**: Scheduled events and room availability

## REST API2 - Decision Support
- **Recommendations**: AHP-based room ranking considering user preferences

## Standards Compliance
- EN 16798-1:2019 (Temperature, Humidity)
- EN 13779:2007 (CO2/Air Quality)
- EN 12464-1:2021 (Lighting)
- WHO Guidelines (Sound Levels)
    """,
    version="1.0.0",
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Serve UI1 at /ui
@app.get("/ui", include_in_schema=False)
def serve_ui1():
    return FileResponse(os.path.join(static_dir, "ui1.html"))


# PYDANTIC MODELS - REST API1 (Data Access)


class Measurement(BaseModel):
    """Sensor measurement data model"""

    ts: str
    room_id: int
    room_name: str
    temperature: float
    humidity: float
    light: int
    sound: int
    air_quality: int
    air_quality_level: str


class RoomFacility(BaseModel):
    """Room facilities data model"""

    room_id: int
    has_projector: bool
    seat_count: int
    has_whiteboard: bool
    has_pc: bool


class CalendarEvent(BaseModel):
    """Calendar event data model"""

    id: int
    room_id: int
    start_time: str
    end_time: str
    event_name: str
    organizer: str


# PYDANTIC MODELS - REST API2 (Decision Support)


class PairwiseComparisons(BaseModel):
    """
    Saaty Scale Pairwise Comparisons for AHP.

    Scale Interpretation:
    - 1: Equal importance
    - 3: Moderate importance
    - 5: Strong importance
    - 7: Very strong importance
    - 9: Extreme importance
    - 2,4,6,8: Intermediate values
    - 1/3, 1/5, etc.: Inverse comparisons
    """

    temp_vs_humidity: float = 1.0
    temp_vs_co2: float = 1.0
    temp_vs_sound: float = 1.0
    temp_vs_light: float = 1.0
    humidity_vs_co2: float = 1.0
    humidity_vs_sound: float = 1.0
    humidity_vs_light: float = 1.0
    co2_vs_sound: float = 1.0
    co2_vs_light: float = 1.0
    sound_vs_light: float = 1.0

    @field_validator("*", mode="before")
    @classmethod
    def validate_positive(cls, v):
        if isinstance(v, (int, float)) and v <= 0:
            raise ValueError("Comparison values must be positive")
        return v


class DesiredProfile(BaseModel):
    """
    Desired environmental profile with EU standard defaults.
    """

    target_temp: float = 21.0  # °C - EN 16798-1 Category II
    target_humidity: float = 45.0  # % - EN 16798-1
    target_co2: int = 800  # ppm - EN 13779 IDA 2
    target_sound: int = 40  # dB - WHO Guidelines
    target_light: int = 500  # lux - EN 12464-1


class UserPreferences(BaseModel):
    """
    Complete user preferences for room recommendation.
    """

    # Hard Constraints
    min_seats: Optional[int] = 0
    needs_projector: Optional[bool] = False
    needs_whiteboard: Optional[bool] = False
    needs_pc: Optional[bool] = False

    # Scheduling Constraints
    desired_date: Optional[str] = None
    needed_duration_minutes: int = 0

    # Soft Constraints (AHP)
    comparisons: PairwiseComparisons = PairwiseComparisons()

    # Desired Profile
    profile: DesiredProfile = DesiredProfile()


class CriteriaDetail(BaseModel):
    """Detailed scoring breakdown for a single criterion"""

    raw_value: float
    target_value: float
    score: float
    standard: str


class RoomRanking(BaseModel):
    """Output model for ranked room recommendations"""

    room_id: int
    room_name: str
    final_score: float
    rank: int
    suggested_start_time: str
    suggested_end_time: str
    ahp_weights: Dict[str, float]
    criteria_scores: Dict[str, float]
    criteria_details: Dict[str, CriteriaDetail]
    consistency_ratio: float
    consistency_acceptable: bool


class AHPWeightsResponse(BaseModel):
    """Response model for AHP weight calculation endpoint"""

    weights: Dict[str, float]
    consistency_ratio: float
    consistency_acceptable: bool
    message: str


# AHP CALCULATOR CLASS


class AHPCalculator:
    """
    Analytic Hierarchy Process (AHP) implementation.
    Based on Thomas Saaty's methodology (1980).
    """

    RI = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }

    CRITERIA_ORDER = ["temperature", "humidity", "co2", "sound", "light"]

    @classmethod
    def build_comparison_matrix(cls, comparisons: PairwiseComparisons) -> np.ndarray:
        """Build the n×n pairwise comparison matrix."""
        n = 5
        matrix = np.ones((n, n))

        matrix[0, 1] = comparisons.temp_vs_humidity
        matrix[0, 2] = comparisons.temp_vs_co2
        matrix[0, 3] = comparisons.temp_vs_sound
        matrix[0, 4] = comparisons.temp_vs_light
        matrix[1, 2] = comparisons.humidity_vs_co2
        matrix[1, 3] = comparisons.humidity_vs_sound
        matrix[1, 4] = comparisons.humidity_vs_light
        matrix[2, 3] = comparisons.co2_vs_sound
        matrix[2, 4] = comparisons.co2_vs_light
        matrix[3, 4] = comparisons.sound_vs_light

        for i in range(n):
            for j in range(i + 1, n):
                matrix[j, i] = 1.0 / matrix[i, j]

        return matrix

    @classmethod
    def calculate_priority_weights(cls, matrix: np.ndarray) -> np.ndarray:
        """Calculate priority weights using normalized column sum method."""
        col_sums = matrix.sum(axis=0)
        normalized = matrix / col_sums
        weights = normalized.mean(axis=1)
        return weights

    @classmethod
    def calculate_consistency_ratio(
        cls, matrix: np.ndarray, weights: np.ndarray
    ) -> float:
        """Calculate Consistency Ratio (CR). CR < 0.10 is acceptable."""
        n = len(weights)
        weighted_sum = matrix @ weights
        lambda_max = np.mean(weighted_sum / weights)
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        RI = cls.RI.get(n, 1.12)
        CR = CI / RI if RI > 0 else 0
        return max(0, CR)

    @classmethod
    def compute_weights(cls, comparisons: PairwiseComparisons) -> tuple:
        """Main method: Compute AHP weights from pairwise comparisons."""
        matrix = cls.build_comparison_matrix(comparisons)
        weights = cls.calculate_priority_weights(matrix)
        cr = cls.calculate_consistency_ratio(matrix, weights)

        weights_dict = {
            name: round(float(w), 4) for name, w in zip(cls.CRITERIA_ORDER, weights)
        }

        return weights_dict, round(cr, 4)


# CRITERIA SCORER CLASS


class CriteriaScorer:
    """Maps raw sensor values to compliance scores (0.0 to 1.0)."""

    @staticmethod
    def score_temperature(value: float, target: float) -> tuple:
        """Standard: EN 16798-1:2019"""
        diff = abs(value - target)
        score = max(0.0, 1.0 - (diff / 5.0))
        return score, "EN 16798-1:2019"

    @staticmethod
    def score_humidity(value: float, target: float) -> tuple:
        """Standard: EN 16798-1:2019"""
        if 40 <= value <= 60:
            score = 1.0
        elif 30 <= value < 40:
            score = 0.6 + (value - 30) / 25
        elif 60 < value <= 70:
            score = 0.6 + (70 - value) / 25
        elif 20 <= value < 30:
            score = 0.3 + (value - 20) / 33
        elif 70 < value <= 80:
            score = 0.3 + (80 - value) / 33
        else:
            score = max(0.0, 0.3 - abs(value - 50) / 100)
        return score, "EN 16798-1:2019"

    @staticmethod
    def score_co2(value: float, target: float) -> tuple:
        """Standard: EN 13779:2007"""
        if value <= target:
            score = 1.0
        elif value >= 1500:
            score = 0.0
        else:
            score = 1.0 - (value - target) / (1500 - target)
        return score, "EN 13779:2007"

    @staticmethod
    def score_sound(value: float, target: float) -> tuple:
        """Standard: WHO Guidelines 1999"""
        if value <= target:
            score = 1.0
        elif value >= 80:
            score = 0.0
        else:
            score = 1.0 - (value - target) / (80 - target)
        return score, "WHO Guidelines 1999"

    @staticmethod
    def score_light(value: float, target: float) -> tuple:
        """Standard: EN 12464-1:2021"""
        if value < 100:
            score = value / 100 * 0.3
        elif value < 300:
            score = 0.3 + (value - 100) / 200 * 0.3
        elif value <= 500:
            score = 0.6 + (value - 300) / 200 * 0.4
        elif value <= 750:
            score = 1.0
        elif value <= 1000:
            score = 1.0 - (value - 750) / 250 * 0.2
        else:
            score = max(0.0, 0.8 - (value - 1000) / 1000 * 0.8)
        return score, "EN 12464-1:2021"


# DATABASE HELPERS


def get_db_connection():
    """Create a database connection with Row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# SCHEDULING HELPERS


def find_slot_on_date(room_id: int, date_str: str, duration_minutes: int):
    """Find an available time slot on a specific date (08:00-20:00)."""
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None, None

    work_start = datetime.combine(target_date, datetime.min.time()).replace(
        hour=8, minute=0
    )
    work_end = datetime.combine(target_date, datetime.min.time()).replace(
        hour=20, minute=0
    )

    now = datetime.now()
    if target_date == now.date():
        if now > work_start:
            work_start = now
        if now >= work_end:
            return None, None
    elif target_date < now.date():
        return None, None

    ws_str = work_start.strftime("%Y-%m-%d %H:%M:%S")
    we_str = work_end.strftime("%Y-%m-%d %H:%M:%S")

    with closing(get_db_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT start_time, end_time FROM calendar 
            WHERE room_id = ? AND end_time > ? AND start_time < ?
            ORDER BY start_time ASC
        """,
            (room_id, ws_str, we_str),
        )
        events = cursor.fetchall()

    current_pointer = work_start

    for event in events:
        evt_start = datetime.strptime(event["start_time"], "%Y-%m-%d %H:%M:%S")
        evt_end = datetime.strptime(event["end_time"], "%Y-%m-%d %H:%M:%S")

        if evt_start > current_pointer:
            gap_minutes = (evt_start - current_pointer).total_seconds() / 60
            if gap_minutes >= duration_minutes:
                return current_pointer, current_pointer + timedelta(
                    minutes=duration_minutes
                )

        if evt_end > current_pointer:
            current_pointer = evt_end

    if current_pointer < work_end:
        final_gap = (work_end - current_pointer).total_seconds() / 60
        if final_gap >= duration_minutes:
            return current_pointer, current_pointer + timedelta(
                minutes=duration_minutes
            )

    return None, None


# DATABASE STARTUP


@app.on_event("startup")
def startup_db_setup():
    """Initialize database tables and insert sample data."""
    with closing(get_db_connection()) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS facilities (
            room_id INTEGER PRIMARY KEY,
            has_projector BOOLEAN,
            seat_count INTEGER,
            has_whiteboard BOOLEAN,
            has_pc BOOLEAN
        )
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS calendar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id INTEGER,
            start_time DATETIME,
            end_time DATETIME,
            event_name TEXT,
            organizer TEXT
        )
        """
        )

        cursor.execute("SELECT COUNT(*) FROM facilities")
        if cursor.fetchone()[0] == 0:
            sample_facilities = [
                (1, 1, 30, 1, 1),
                (2, 1, 20, 1, 0),
                (3, 0, 15, 1, 0),
            ]
            cursor.executemany(
                "INSERT INTO facilities VALUES (?, ?, ?, ?, ?)", sample_facilities
            )
            print("Inserted sample facilities data")

        cursor.execute("SELECT COUNT(*) FROM calendar")
        if cursor.fetchone()[0] == 0:
            now = datetime.now()
            sample_events = [
                (
                    1,
                    now.strftime("%Y-%m-%d 09:00:00"),
                    now.strftime("%Y-%m-%d 11:00:00"),
                    "IoT Lecture",
                    "Prof. Kubler",
                ),
                (
                    1,
                    now.strftime("%Y-%m-%d 14:00:00"),
                    now.strftime("%Y-%m-%d 16:00:00"),
                    "Lab Session",
                    "TA Smith",
                ),
                (
                    2,
                    now.strftime("%Y-%m-%d 10:00:00"),
                    now.strftime("%Y-%m-%d 12:00:00"),
                    "Math Class",
                    "Prof. Johnson",
                ),
            ]
            cursor.executemany(
                """
                INSERT INTO calendar (room_id, start_time, end_time, event_name, organizer) 
                VALUES (?, ?, ?, ?, ?)
            """,
                sample_events,
            )
            print("Inserted sample calendar events")

        conn.commit()


# REST API ENDPOINTS - ROOT


@app.get("/", include_in_schema=False)
def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


# REST API1 - DATA ACCESS ENDPOINTS


@app.get(
    "/rooms/{room_id}/measurements",
    response_model=List[Measurement],
    tags=["REST API1 - Sensors"],
    summary="Get sensor measurements for a room",
)
def get_sensor_data(
    room_id: int,
    start_time: Optional[str] = Query(None, description="Format: YYYY-MM-DD HH:MM:SS"),
    end_time: Optional[str] = Query(None, description="Format: YYYY-MM-DD HH:MM:SS"),
    limit: int = Query(100, description="Maximum records to return"),
):
    """Get temperature, CO2, sound, humidity, light values over time."""
    query = "SELECT * FROM measurements WHERE room_id = ?"
    params = [room_id]

    if start_time:
        query += " AND ts >= ?"
        params.append(start_time)

    if end_time:
        query += " AND ts <= ?"
        params.append(end_time)

    query += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)

    with closing(get_db_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404, detail=f"No measurements found for room {room_id}"
        )

    return [dict(row) for row in rows]


@app.get(
    "/rooms/{room_id}/facilities",
    response_model=RoomFacility,
    tags=["REST API1 - Facilities"],
    summary="Get facilities for a room",
)
def get_room_facilities(room_id: int):
    """Get projector, seats, whiteboard, PC availability."""
    with closing(get_db_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM facilities WHERE room_id = ?", (room_id,))
        row = cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=404, detail=f"Facilities not found for room {room_id}"
        )

    return dict(row)


@app.get(
    "/rooms/facilities",
    response_model=List[RoomFacility],
    tags=["REST API1 - Facilities"],
    summary="Get facilities for all rooms",
)
def get_all_facilities():
    """Get all room facilities."""
    with closing(get_db_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM facilities ORDER BY room_id")
        rows = cursor.fetchall()

    return [dict(row) for row in rows]


@app.get(
    "/rooms/{room_id}/calendar",
    response_model=List[CalendarEvent],
    tags=["REST API1 - Calendar"],
    summary="Get calendar events for a room",
)
def get_calendar_events(
    room_id: int,
    start_date: Optional[str] = Query(None, description="Format: YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Format: YYYY-MM-DD"),
):
    """Get scheduled events for a room."""
    query = "SELECT * FROM calendar WHERE room_id = ?"
    params = [room_id]

    if start_date:
        query += " AND DATE(start_time) >= ?"
        params.append(start_date)

    if end_date:
        query += " AND DATE(end_time) <= ?"
        params.append(end_date)

    query += " ORDER BY start_time ASC"

    with closing(get_db_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    return [dict(row) for row in rows]


@app.get(
    "/calendar",
    response_model=List[CalendarEvent],
    tags=["REST API1 - Calendar"],
    summary="Get all calendar events",
)
def get_all_calendar_events(
    start_date: Optional[str] = Query(None, description="Format: YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Format: YYYY-MM-DD"),
):
    """Get all events across all rooms."""
    query = "SELECT * FROM calendar WHERE 1=1"
    params = []

    if start_date:
        query += " AND DATE(start_time) >= ?"
        params.append(start_date)

    if end_date:
        query += " AND DATE(end_time) <= ?"
        params.append(end_date)

    query += " ORDER BY start_time ASC"

    with closing(get_db_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    return [dict(row) for row in rows]


# REST API2 - DECISION SUPPORT ENDPOINTS


@app.post(
    "/ahp/weights",
    response_model=AHPWeightsResponse,
    tags=["REST API2 - Decision Support"],
    summary="Calculate AHP weights from pairwise comparisons",
)
def calculate_ahp_weights(comparisons: PairwiseComparisons):
    """Calculate priority weights using AHP. CR should be < 0.10."""
    weights, cr = AHPCalculator.compute_weights(comparisons)
    is_consistent = cr < 0.10

    message = (
        "Comparisons are consistent." if is_consistent else f"Warning: CR={cr} > 0.10"
    )

    return AHPWeightsResponse(
        weights=weights,
        consistency_ratio=cr,
        consistency_acceptable=is_consistent,
        message=message,
    )


@app.post(
    "/recommendations",
    response_model=List[RoomRanking],
    tags=["REST API2 - Decision Support"],
    summary="Get ranked room recommendations",
)
def recommend_rooms(prefs: UserPreferences):
    """
    Get room recommendations using AHP-based MCDM.

    Filters by hard constraints, ranks by weighted criteria scores.
    """
    ahp_weights, consistency_ratio = AHPCalculator.compute_weights(prefs.comparisons)
    consistency_ok = consistency_ratio < 0.10

    scorer = CriteriaScorer()
    profile = prefs.profile
    ranked_rooms = []

    with closing(get_db_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM facilities")
        facilities = cursor.fetchall()

        for room in facilities:
            room_id = room["room_id"]

            # Hard Constraints
            if prefs.needs_projector and not room["has_projector"]:
                continue
            if prefs.needs_whiteboard and not room["has_whiteboard"]:
                continue
            if prefs.needs_pc and not room["has_pc"]:
                continue
            if room["seat_count"] < prefs.min_seats:
                continue

            # time slot
            suggested_start = "Available Now"
            suggested_end = "Flexible"

            if prefs.desired_date and prefs.needed_duration_minutes > 0:
                start_dt, end_dt = find_slot_on_date(
                    room_id, prefs.desired_date, prefs.needed_duration_minutes
                )
                if start_dt is None:
                    continue
                suggested_start = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                suggested_end = end_dt.strftime("%Y-%m-%d %H:%M:%S")

            # grab sensor data
            cursor.execute(
                """
                SELECT * FROM measurements 
                WHERE room_id = ? ORDER BY ts DESC LIMIT 1
            """,
                (room_id,),
            )
            data = cursor.fetchone()

            if not data:
                continue

            # Score Criteria
            temp_score, temp_std = scorer.score_temperature(
                data["temperature"], profile.target_temp
            )
            humidity_score, humidity_std = scorer.score_humidity(
                data["humidity"], profile.target_humidity
            )
            co2_score, co2_std = scorer.score_co2(
                data["air_quality"], profile.target_co2
            )
            sound_score, sound_std = scorer.score_sound(
                data["sound"], profile.target_sound
            )
            light_score, light_std = scorer.score_light(
                data["light"], profile.target_light
            )

            criteria_scores = {
                "temperature": round(temp_score, 3),
                "humidity": round(humidity_score, 3),
                "co2": round(co2_score, 3),
                "sound": round(sound_score, 3),
                "light": round(light_score, 3),
            }

            criteria_details = {
                "temperature": CriteriaDetail(
                    raw_value=data["temperature"],
                    target_value=profile.target_temp,
                    score=round(temp_score, 3),
                    standard=temp_std,
                ),
                "humidity": CriteriaDetail(
                    raw_value=data["humidity"],
                    target_value=profile.target_humidity,
                    score=round(humidity_score, 3),
                    standard=humidity_std,
                ),
                "co2": CriteriaDetail(
                    raw_value=data["air_quality"],
                    target_value=profile.target_co2,
                    score=round(co2_score, 3),
                    standard=co2_std,
                ),
                "sound": CriteriaDetail(
                    raw_value=data["sound"],
                    target_value=profile.target_sound,
                    score=round(sound_score, 3),
                    standard=sound_std,
                ),
                "light": CriteriaDetail(
                    raw_value=data["light"],
                    target_value=profile.target_light,
                    score=round(light_score, 3),
                    standard=light_std,
                ),
            }

            final_score = sum(
                criteria_scores[c] * ahp_weights[c] for c in criteria_scores
            )

            ranked_rooms.append(
                {
                    "room_id": room_id,
                    "room_name": data["room_name"],
                    "final_score": round(final_score * 100, 1),
                    "rank": 0,
                    "suggested_start_time": suggested_start,
                    "suggested_end_time": suggested_end,
                    "ahp_weights": ahp_weights,
                    "criteria_scores": criteria_scores,
                    "criteria_details": criteria_details,
                    "consistency_ratio": consistency_ratio,
                    "consistency_acceptable": consistency_ok,
                }
            )

    ranked_rooms.sort(key=lambda x: x["final_score"], reverse=True)
    for i, room in enumerate(ranked_rooms):
        room["rank"] = i + 1

    return ranked_rooms


# ADMIN ENDPOINTS (for UI2)


@app.get(
    "/admin/sensor-stats/{room_id}",
    tags=["Admin - Monitoring"],
    summary="Get sensor statistics for monitoring",
)
def get_sensor_stats(
    room_id: int, hours: int = Query(24, description="Hours to analyze")
):
    """Get aggregated sensor stats for admin dashboard."""
    cutoff = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

    with closing(get_db_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_readings,
                AVG(temperature) as avg_temp,
                MIN(temperature) as min_temp,
                MAX(temperature) as max_temp,
                AVG(humidity) as avg_humidity,
                AVG(air_quality) as avg_co2,
                AVG(sound) as avg_sound,
                AVG(light) as avg_light,
                MIN(ts) as first_reading,
                MAX(ts) as last_reading
            FROM measurements 
            WHERE room_id = ? AND ts >= ?
        """,
            (room_id, cutoff),
        )
        row = cursor.fetchone()

    if not row or row["total_readings"] == 0:
        raise HTTPException(status_code=404, detail=f"No data for room {room_id}")

    return {
        "room_id": room_id,
        "period_hours": hours,
        "total_readings": row["total_readings"],
        "temperature": {
            "avg": round(row["avg_temp"], 1) if row["avg_temp"] else None,
            "min": round(row["min_temp"], 1) if row["min_temp"] else None,
            "max": round(row["max_temp"], 1) if row["max_temp"] else None,
        },
        "humidity": {
            "avg": round(row["avg_humidity"], 1) if row["avg_humidity"] else None
        },
        "co2": {"avg": round(row["avg_co2"], 0) if row["avg_co2"] else None},
        "sound": {"avg": round(row["avg_sound"], 0) if row["avg_sound"] else None},
        "light": {"avg": round(row["avg_light"], 0) if row["avg_light"] else None},
        "first_reading": row["first_reading"],
        "last_reading": row["last_reading"],
    }


@app.get(
    "/admin/system-status",
    tags=["Admin - Monitoring"],
    summary="Get system status overview",
)
def get_system_status():
    """Get overall system health."""
    with closing(get_db_connection()) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(DISTINCT room_id) FROM facilities")
        room_count = cursor.fetchone()[0]

        one_hour_ago = (datetime.now() - timedelta(hours=1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        cursor.execute(
            "SELECT COUNT(*) FROM measurements WHERE ts >= ?", (one_hour_ago,)
        )
        recent_measurements = cursor.fetchone()[0]

        cursor.execute(
            "SELECT DISTINCT room_id FROM measurements WHERE ts >= ?", (one_hour_ago,)
        )
        active_rooms = [row[0] for row in cursor.fetchall()]

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("SELECT COUNT(*) FROM calendar WHERE start_time >= ?", (now,))
        upcoming_events = cursor.fetchone()[0]

    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "total_rooms": room_count,
        "active_rooms_last_hour": active_rooms,
        "measurements_last_hour": recent_measurements,
        "upcoming_calendar_events": upcoming_events,
        "database": DB_PATH,
    }


# entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
