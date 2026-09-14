"""
backend/analytics/progress_tracker.py
====================================
Lightweight analytics tracking for the Smart AI Learning Platform.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

ANALYTICS_FILE = os.path.join("data", "analytics_data.json")

def _load_data() -> Dict[str, Any]:
    initial_structure = {
        "stats": {
            "total_questions": 0,
            "total_quizzes": 0,
            "average_quiz_score": 0,
            "documents_indexed": 0
        },
        "questions": [],
        "quizzes": [],
        "topic_mastery": {}
    }

    if not os.path.exists(ANALYTICS_FILE):
        os.makedirs(os.path.dirname(ANALYTICS_FILE), exist_ok=True)
        _save_data(initial_structure)
        return initial_structure
    
    try:
        with open(ANALYTICS_FILE, "r") as f:
            data = json.load(f)
            
            # Ensure all expected keys exist
            if not isinstance(data, dict):
                data = initial_structure
            
            if "questions" not in data:
                data["questions"] = []
            if "quizzes" not in data:
                data["quizzes"] = []
            
            # Migration & Cleanup
            needs_save = False
            
            # 1. Migrate questions
            for q in data["questions"]:
                topic = q.get("topic")
                if not topic or topic.strip() in ("", "unknown"):
                    q["topic"] = "General"
                    needs_save = True
            
            # 2. Migrate quizzes
            for q in data["quizzes"]:
                topic = q.get("topic")
                if not topic or topic.strip() in ("", "unknown"):
                    q["topic"] = "General"
                    needs_save = True
                
                if "percentage" not in q:
                    score = q.get("score", 0)
                    total = q.get("total", 1)
                    q["percentage"] = (score / total) * 100 if total > 0 else 0
                    needs_save = True

            # 3. Dynamically re-build stats and topic_mastery to ensure perfect alignment
            # Build stats
            try:
                from backend.utils.indexer import get_indexed_files
                docs_count = len(get_indexed_files())
            except Exception:
                docs_count = data.get("stats", {}).get("documents_indexed", 0)

            all_percentages = [q["percentage"] for q in data["quizzes"]]
            avg_score = round(sum(all_percentages) / len(all_percentages), 1) if all_percentages else 0.0

            # Get last quiz score string
            last_quiz = data["quizzes"][-1] if data["quizzes"] else None
            last_score_str = f"{last_quiz['score']}/{last_quiz['total']}" if last_quiz else ""

            old_stats = data.get("stats", {})
            new_stats = {
                "total_questions": len(data["questions"]),
                "total_quizzes": len(data["quizzes"]),
                "average_quiz_score": avg_score,
                "documents_indexed": docs_count,
                "last_quiz_score": last_score_str
            }
            if old_stats != new_stats:
                data["stats"] = new_stats
                needs_save = True

            # Build topic_mastery
            new_topic_mastery = {}
            for q in data["questions"]:
                t = q["topic"]
                if t not in new_topic_mastery:
                    new_topic_mastery[t] = {"questions": 0, "quiz_avg": 0, "quiz_count": 0, "_quiz_pcts": []}
                new_topic_mastery[t]["questions"] += 1

            for q in data["quizzes"]:
                t = q["topic"]
                if t not in new_topic_mastery:
                    new_topic_mastery[t] = {"questions": 0, "quiz_avg": 0, "quiz_count": 0, "_quiz_pcts": []}
                new_topic_mastery[t]["_quiz_pcts"].append(q["percentage"])
                new_topic_mastery[t]["quiz_count"] += 1

            for t, tstats in new_topic_mastery.items():
                pcts = tstats.pop("_quiz_pcts")
                tstats["quiz_avg"] = round(sum(pcts) / len(pcts), 1) if pcts else 0.0

            if data.get("topic_mastery") != new_topic_mastery:
                data["topic_mastery"] = new_topic_mastery
                needs_save = True

            if needs_save:
                _save_data(data)
                
            return data
    except (json.JSONDecodeError, IOError):
        return initial_structure

def _save_data(data: Dict[str, Any]):
    with open(ANALYTICS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def record_question(question: str, topic: str, sources: List[str], mode: str):
    data = _load_data()
    
    # Normalize topic
    if not topic or topic.strip() in ("", "unknown"):
        topic = "General"
        
    data["stats"]["total_questions"] += 1
    
    data["questions"].append({
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "topic": topic,
        "mode": mode
    })
    
    # Update topic frequency
    data["topic_mastery"][topic] = data["topic_mastery"].get(topic, {"questions": 0, "quiz_avg": 0, "quiz_count": 0})
    data["topic_mastery"][topic]["questions"] += 1
    
    _save_data(data)
    logger.info(f"Recorded question about {topic}")

def record_quiz_result(topic: str, score: int, total: int, quiz_type: str):
    data = _load_data()
    
    # Normalize topic
    if not topic or topic.strip() in ("", "unknown"):
        topic = "General"
        
    data["stats"]["total_quizzes"] += 1
    
    percentage = (score / total) * 100 if total > 0 else 0
    data["quizzes"].append({
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "score": score,
        "total": total,
        "percentage": percentage,
        "type": quiz_type
    })
    
    # Update global average
    all_percentages = [q["percentage"] for q in data["quizzes"]]
    data["stats"]["average_quiz_score"] = round(sum(all_percentages) / len(all_percentages), 1)
    
    # Update topic mastery
    topic_data = data["topic_mastery"].get(topic, {"questions": 0, "quiz_avg": 0, "quiz_count": 0})
    current_avg = topic_data["quiz_avg"]
    count = topic_data["quiz_count"]
    new_avg = (current_avg * count + percentage) / (count + 1)
    topic_data["quiz_avg"] = round(new_avg, 1)
    topic_data["quiz_count"] += 1
    data["topic_mastery"][topic] = topic_data
    
    _save_data(data)
    logger.info(f"Recorded quiz result for {topic}: {score}/{total}")

def record_document_indexed(count: int = 1):
    data = _load_data()
    data["stats"]["documents_indexed"] += count
    _save_data(data)

def get_dashboard_data() -> Dict[str, Any]:
    # Running _load_data() automatically triggers migration, normalization, and stats/topic_mastery syncing!
    data = _load_data()
    
    # Force documents_indexed to be dynamically accurate from Chroma
    try:
        from backend.utils.indexer import get_indexed_files
        data["stats"]["documents_indexed"] = len(get_indexed_files())
    except Exception as exc:
        logger.error(f"Failed to get dynamic document count for dashboard: {exc}")
    
    # Format for frontend
    topic_summary = []
    for topic, stats in data["topic_mastery"].items():
        mastery = (stats["quiz_avg"] * 0.7 + min(stats["questions"] * 5, 30)) # Mix of quiz and volume
        topic_summary.append({
            "topic": topic,
            "mastery_score": round(min(mastery, 100), 1),
            "question_count": stats["questions"]
        })
    
    recent_activity = []
    # Combine last 5 questions and quizzes
    combined = []
    for q in data["questions"][-5:]:
        combined.append({"type": "question", "timestamp": q["timestamp"], "topic": q["topic"]})
    for q in data["quizzes"][-5:]:
        combined.append({"type": "quiz", "timestamp": q["timestamp"], "topic": q["topic"]})
    
    combined.sort(key=lambda x: x["timestamp"], reverse=True)
    recent_activity = combined[:5]
    
    return {
        "stats": data["stats"],
        "topic_summary": topic_summary,
        "recent_activity": recent_activity
    }


def reset_analytics() -> Dict[str, Any]:
    """Reset all analytics and progress data back to initial state."""
    initial_structure = {
        "stats": {
            "total_questions": 0,
            "total_quizzes": 0,
            "average_quiz_score": 0,
            "documents_indexed": 0
        },
        "questions": [],
        "quizzes": [],
        "topic_mastery": {}
    }
    _save_data(initial_structure)
    logger.info("Analytics data has been reset to initial state.")
    return initial_structure

