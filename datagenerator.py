"""
Synthetic Data Generator: Simulates realistic student study habits over 30 days.
Generates patterns conforming to Slide 5 & Slide 8 (e.g., peak distraction around 8-9 PM).
"""

import random
from datetime import datetime, timedelta, date
import pandas as pd
from data_manager import save_session, CSV_PATH, ensure_data_dir
from focus import calculate_focus_score

SUBJECTS = ["Mathematics", "Computer Science", "Physics", "Chemistry", "English Literature", "History"]

def generate_sample_data(num_days: int = 30) -> None:
    """Populate data/study_sessions.csv with realistic student study logs."""
    ensure_data_dir()
    
    # Start fresh with empty file
    if pd.io.common.file_exists(CSV_PATH):
        import os
        os.remove(CSV_PATH)
    ensure_data_dir()

    today = date.today()
    random.seed(42)  # reproducible

    for day_offset in range(num_days - 1, -1, -1):
        curr_date = today - timedelta(days=day_offset)
        day_of_week = curr_date.strftime("%A")
        
        # 1 to 3 study sessions per day
        num_sessions = random.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]
        
        # Distribute hours across common study times: morning (9-11), afternoon (14-17), evening (19-22)
        possible_hours = [9, 11, 14, 16, 18, 19, 20, 21, 22]
        chosen_hours = sorted(random.sample(possible_hours, num_sessions))

        for hr in chosen_hours:
            subject = random.choice(SUBJECTS)
            total_time = random.choice([45, 60, 90, 120])
            
            # If time is 8 PM (20) or 9 PM (21), high likelihood of phone & social distraction (Slide 8)
            is_evening_slump = hr in [20, 21]
            is_late_night = hr >= 22
            
            if is_evening_slump:
                # 8-9 PM: phone distraction is high
                studying_mins = total_time * random.uniform(0.40, 0.60)
                phone_mins = total_time * random.uniform(0.20, 0.35)
                talking_mins = total_time * random.uniform(0.05, 0.15)
                break_mins = total_time - (studying_mins + phone_mins + talking_mins)
                social_mins = random.uniform(0, 10)
                wandering_mins = random.uniform(0, 10)
            elif is_late_night:
                studying_mins = total_time * random.uniform(0.45, 0.65)
                phone_mins = total_time * random.uniform(0.15, 0.25)
                talking_mins = total_time * random.uniform(0.02, 0.08)
                break_mins = total_time - (studying_mins + phone_mins + talking_mins)
                social_mins = random.uniform(0, 8)
                wandering_mins = random.uniform(5, 15)
            else:
                # Morning / Afternoon: higher focus
                studying_mins = total_time * random.uniform(0.70, 0.85)
                phone_mins = total_time * random.uniform(0.02, 0.10)
                talking_mins = total_time * random.uniform(0.00, 0.05)
                break_mins = total_time - (studying_mins + phone_mins + talking_mins)
                social_mins = 0.0
                wandering_mins = random.uniform(0, 5)

            # Round minutes
            studying_mins = round(max(5.0, studying_mins), 1)
            phone_mins = round(max(0.0, phone_mins), 1)
            talking_mins = round(max(0.0, talking_mins), 1)
            break_mins = round(max(0.0, break_mins), 1)
            social_mins = round(max(0.0, social_mins), 1)
            wandering_mins = round(max(0.0, wandering_mins), 1)
            total_time = studying_mins + phone_mins + talking_mins + break_mins + social_mins + wandering_mins

            breakdown = {
                "studying": studying_mins,
                "phone": phone_mins,
                "talking": talking_mins,
                "break": break_mins,
                "social_media": social_mins,
                "wandering": wandering_mins
            }
            res = calculate_focus_score(breakdown)

            sess_timestamp = datetime(curr_date.year, curr_date.month, curr_date.day, hr, random.randint(0, 45))
            
            session_dict = {
                "session_id": f"sess_{int(sess_timestamp.timestamp())}",
                "timestamp": sess_timestamp.isoformat(),
                "date": str(curr_date),
                "hour_of_day": hr,
                "day_of_week": day_of_week,
                "subject": subject,
                "total_minutes": round(total_time, 1),
                "studying_minutes": studying_mins,
                "phone_minutes": phone_mins,
                "talking_minutes": talking_mins,
                "break_minutes": break_mins,
                "social_media_minutes": social_mins,
                "wandering_minutes": wandering_mins,
                "focus_score": res["focus_score"],
                "study_ratio_pct": res["study_ratio_pct"],
                "biggest_distraction": res["biggest_distraction"]
            }
            save_session(session_dict)

    # Insert today's showcase session (Exact Slide 5 data: 120m, 70 study, 20 phone, 10 talking, 20 break)
    slide5_session = {
        "session_id": f"sess_showcase_{int(datetime.now().timestamp())}",
        "timestamp": datetime.combine(today, datetime.min.time()).replace(hour=20, minute=0).isoformat(),
        "date": str(today),
        "hour_of_day": 20,
        "day_of_week": today.strftime("%A"),
        "subject": "Mathematics (Calculus)",
        "total_minutes": 120.0,
        "studying_minutes": 70.0,
        "phone_minutes": 20.0,
        "talking_minutes": 10.0,
        "break_minutes": 20.0,
        "social_media_minutes": 0.0,
        "wandering_minutes": 0.0,
        "focus_score": 75,
        "study_ratio_pct": 58.3,
        "biggest_distraction": "Phone"
    }
    save_session(slide5_session)
    print(f"Generated synthetic study dataset with {num_days} days of history.")


if __name__ == "__main__":
    generate_sample_data()
