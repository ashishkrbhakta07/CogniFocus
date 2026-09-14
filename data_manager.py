"""
Data Manager: CSV Persistence and Aggregation for Study Sessions
Handles loading, saving, querying, and aggregating study records.
"""

import os
from datetime import datetime, date
from typing import Dict, Any, List, Optional
import pandas as pd

# Path to the data directory and CSV database
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "study_sessions.csv")

COLUMNS = [
    "session_id",
    "timestamp",
    "date",
    "hour_of_day",
    "day_of_week",
    "subject",
    "total_minutes",
    "studying_minutes",
    "phone_minutes",
    "talking_minutes",
    "break_minutes",
    "social_media_minutes",
    "wandering_minutes",
    "focus_score",
    "study_ratio_pct",
    "biggest_distraction",
    "high_distraction_flag"  # 1 if focus_score < 70 or distraction > 25 mins, else 0
]


def ensure_data_dir() -> None:
    """1. Ensure data directory and CSV file exist with correct headers."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(CSV_PATH, index=False)


def load_sessions() -> pd.DataFrame:
    """2. Load all study sessions from CSV as a Pandas DataFrame."""
    ensure_data_dir()
    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return pd.DataFrame(columns=COLUMNS)
        # Type conversions
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour_of_day"] = df["hour_of_day"].astype(int)
        df["focus_score"] = df["focus_score"].astype(int)
        return df
    except Exception as e:
        print(f"Error loading sessions: {e}")
        return pd.DataFrame(columns=COLUMNS)


def save_session(session: Dict[str, Any]) -> None:
    """3. Append a new study session record to CSV."""
    ensure_data_dir()
    df = load_sessions()

    # Determine high distraction flag for ML target
    distraction_total = (
        session.get("phone_minutes", 0) +
        session.get("talking_minutes", 0) +
        session.get("social_media_minutes", 0) +
        session.get("wandering_minutes", 0)
    )
    is_high_distraction = 1 if (session.get("focus_score", 100) < 70 or distraction_total >= 25) else 0

    record = {
        "session_id": session.get("session_id", f"sess_{int(datetime.now().timestamp())}"),
        "timestamp": session.get("timestamp", datetime.now().isoformat()),
        "date": session.get("date", str(date.today())),
        "hour_of_day": int(session.get("hour_of_day", datetime.now().hour)),
        "day_of_week": session.get("day_of_week", datetime.now().strftime("%A")),
        "subject": session.get("subject", "General Study"),
        "total_minutes": float(session.get("total_minutes", 0)),
        "studying_minutes": float(session.get("studying_minutes", 0)),
        "phone_minutes": float(session.get("phone_minutes", 0)),
        "talking_minutes": float(session.get("talking_minutes", 0)),
        "break_minutes": float(session.get("break_minutes", 0)),
        "social_media_minutes": float(session.get("social_media_minutes", 0)),
        "wandering_minutes": float(session.get("wandering_minutes", 0)),
        "focus_score": int(session.get("focus_score", 0)),
        "study_ratio_pct": float(session.get("study_ratio_pct", 0.0)),
        "biggest_distraction": session.get("biggest_distraction", "None"),
        "high_distraction_flag": is_high_distraction
    }

    new_row_df = pd.DataFrame([record])
    if df.empty:
        new_row_df.to_csv(CSV_PATH, index=False)
    else:
        new_row_df.to_csv(CSV_PATH, mode="a", header=False, index=False)


def get_today_summary() -> Dict[str, Any]:
    """4. Get aggregated metrics for today's sessions."""
    df = load_sessions()
    today_str = str(date.today())
    today_df = df[df["date"] == today_str] if not df.empty else pd.DataFrame()

    if today_df.empty:
        # Fallback to the Slide 5 benchmark if no session logged today yet
        return {
            "focus_score": 75,
            "total_study_mins": 70,
            "total_distraction_mins": 30,
            "total_break_mins": 20,
            "total_session_mins": 120,
            "active_ratio_pct": 58.3,
            "top_distraction": "Phone",
            "sessions_count": 1,
            "is_sample": True
        }

    total_study = today_df["studying_minutes"].sum()
    total_break = today_df["break_minutes"].sum()
    total_distraction = (
        today_df["phone_minutes"].sum() +
        today_df["talking_minutes"].sum() +
        today_df["social_media_minutes"].sum() +
        today_df["wandering_minutes"].sum()
    )
    total_time = today_df["total_minutes"].sum()
    avg_score = int(today_df["focus_score"].mean())
    active_ratio = (total_study / total_time * 100.0) if total_time > 0 else 0.0

    # Find the top distraction
    distractions = {
        "Phone": today_df["phone_minutes"].sum(),
        "Talking": today_df["talking_minutes"].sum(),
        "Social Media": today_df["social_media_minutes"].sum(),
        "Mind Wandering": today_df["wandering_minutes"].sum(),
    }
    top_dist = max(distractions, key=distractions.get)
    if distractions[top_dist] == 0:
        top_dist = "None"

    return {
        "focus_score": avg_score,
        "total_study_mins": round(total_study, 1),
        "total_distraction_mins": round(total_distraction, 1),
        "total_break_mins": round(total_break, 1),
        "total_session_mins": round(total_time, 1),
        "active_ratio_pct": round(active_ratio, 1),
        "top_distraction": top_dist,
        "sessions_count": len(today_df),
        "is_sample": False
    }


def get_activity_breakdown(df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """5. Return dictionary mapping activity names to total minutes (for the donut chart)."""
    if df is None:
        df = load_sessions()
    if df.empty:
        # Default Slide 5 example
        return {
            "Studying": 70.0,
            "Phone": 20.0,
            "Talking": 10.0,
            "Break": 20.0
        }
    return {
        "Studying": float(df["studying_minutes"].sum()),
        "Phone": float(df["phone_minutes"].sum()),
        "Talking": float(df["talking_minutes"].sum()),
        "Social Media": float(df["social_media_minutes"].sum()),
        "Mind Wandering": float(df["wandering_minutes"].sum()),
        "Break": float(df["break_minutes"].sum()),
    }