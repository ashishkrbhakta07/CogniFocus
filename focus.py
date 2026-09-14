"""
Focus Engine: Mathematical Scoring Model & Feedback Generator
Calculates Focus Score (0-100), efficiency metrics, and rule-based diagnostic insights.
Based on AI Study Distraction Detector specs (Slide 5 & Slide 6).
"""

from typing import Dict, Any, Tuple

# Penalty weights for different distractions
DISTRACTION_WEIGHTS = {
    "phone": 1.2,          # High cognitive switching penalty
    "social_media": 1.3,   # Very high immersion penalty
    "talking": 0.9,        # Moderate disruption
    "wandering": 0.8,      # Internal mind-wandering
    "gaming": 1.4,         # High immersion penalty
    "other_distraction": 1.0
}

def calculate_focus_score(activities: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate focus score based on activity durations in minutes.
    
    Formula Rules:
    1. Productive study time increases the score.
    2. Unplanned distraction time aggressively lowers the score.
    3. Short intentional breaks (e.g., Pomodoro <= 10m) are protected.
    4. Long/excessive breaks (> 15m) are penalized.
    
    Example (Slide 5 & 6):
    - 70 min studying, 20 min phone, 10 min talking, 20 min break (Total: 120 min)
    - Active Study Time: 58.3%
    - Focus Score: ~75/100
    """
    study_time = float(activities.get("studying", 0.0))
    phone_time = float(activities.get("phone", 0.0))
    talking_time = float(activities.get("talking", 0.0))
    break_time = float(activities.get("break", 0.0))
    social_time = float(activities.get("social_media", 0.0))
    wandering_time = float(activities.get("wandering", 0.0))
    gaming_time = float(activities.get("gaming", 0.0))

    total_time = (study_time + phone_time + talking_time + break_time + 
                  social_time + wandering_time + gaming_time)

    if total_time <= 0:
        return {
            "focus_score": 0,
            "study_ratio_pct": 0.0,
            "total_minutes": 0,
            "active_minutes": 0,
            "distraction_minutes": 0,
            "break_minutes": 0,
            "biggest_distraction": "None",
            "ai_insight": "No activity logged yet. Start studying to build your score!",
            "suggested_action": "Begin with a 25-minute distraction-free study block.",
            "status_label": "Not Started"
        }

    # Pomodoro & cognitive science guidelines: up to 15-20 mins break in a 2-hour window is healthy
    healthy_break_allowance = max(10.0, (study_time / 50.0) * 12.0)
    intentional_break = min(break_time, healthy_break_allowance)
    excessive_break = max(0.0, break_time - healthy_break_allowance)

    # Active study ratio
    study_ratio = (study_time / total_time) * 100.0

    # Distraction penalties
    unplanned_distractions = (phone_time * DISTRACTION_WEIGHTS["phone"] +
                             social_time * DISTRACTION_WEIGHTS["social_media"] +
                             gaming_time * DISTRACTION_WEIGHTS["gaming"] +
                             talking_time * DISTRACTION_WEIGHTS["talking"] +
                             wandering_time * DISTRACTION_WEIGHTS["wandering"] +
                             excessive_break * 1.0)

    # Effective productive credit: 1.0 for study, 0.70 for intentional recovery break
    effective_productive_time = study_time + (intentional_break * 0.70)
    base_score = (effective_productive_time / total_time) * 100.0

    # Distraction deduction
    distraction_penalty = (unplanned_distractions / total_time) * 20.0

    # Final calibrated score
    raw_score = base_score - distraction_penalty + 10.0
    final_score = int(round(max(0.0, min(100.0, raw_score))))

    # Total distraction minutes
    total_distraction_mins = phone_time + social_time + gaming_time + talking_time + wandering_time + excessive_break

    # Determine biggest distraction
    distraction_map = {
        "Phone": phone_time,
        "Social Media": social_time,
        "Talking / Interruptions": talking_time,
        "Mind Wandering": wandering_time,
        "Gaming": gaming_time,
        "Excessive Break": excessive_break
    }
    top_distraction_name = max(distraction_map, key=distraction_map.get)
    top_distraction_val = distraction_map[top_distraction_name]
    
    if top_distraction_val <= 0:
        top_distraction_name = "None"

    # Generate insights and suggested actions (Matching Slide 6 & 7)
    ai_insight, suggested_action, status_label = _generate_insights(
        final_score, study_ratio, top_distraction_name, top_distraction_val, total_time
    )

    return {
        "focus_score": final_score,
        "study_ratio_pct": round(study_ratio, 1),
        "total_minutes": round(total_time, 1),
        "active_minutes": round(study_time, 1),
        "distraction_minutes": round(total_distraction_mins, 1),
        "break_minutes": round(break_time, 1),
        "biggest_distraction": top_distraction_name,
        "ai_insight": ai_insight,
        "suggested_action": suggested_action,
        "status_label": status_label
    }


def _generate_insights(score: int, study_ratio: float, top_distraction: str, 
                       top_distraction_val: float, total_time: float) -> Tuple[str, str, str]:
    """Generates personalized AI coach insights and actionable advice."""
    
    if score >= 85:
        status_label = "Deep Flow State"
        insight = f"Outstanding session! You maintained productive focus for {study_ratio:.0f}% of the time."
        action = "Maintain this rhythm. Take a 5-minute hydration break before your next block."
    elif score >= 70:
        status_label = "Good Focus"
        if top_distraction != "None":
            insight = f"Your focus was strong, but {top_distraction.lower()} interruptions ({top_distraction_val:.0f}m) reduced your score."
            action = f"Keep your {top_distraction.lower()} in another room during your first 25-min block."
        else:
            insight = "Solid study effort with steady focus throughout."
            action = "Try shortening breaks to under 10 minutes to sustain cognitive momentum."
    elif score >= 50:
        status_label = "Moderate Distraction"
        insight = f"Over {(100 - study_ratio):.0f}% of session time was spent off-task. Major leak: {top_distraction}."
        action = f"Use 'Do Not Disturb' mode and commit to one single objective for the next 30 minutes."
    else:
        status_label = "High Distraction"
        insight = f"Severe focus fragmentation detected. Only {study_ratio:.0f}% was active study time."
        action = "Reset your environment. Step away for a 10-minute walk and restart with a 15-minute micro-sprint."

    return insight, action, status_label
