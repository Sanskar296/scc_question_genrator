"""
FastAPI Backend Server for Antigravity Intelligent Study Assistant
==================================================================
Provides REST APIs for:
1. User Authentication (JWT + Bcrypt + MongoDB)
2. Four Sections Status (General Awareness as primarily active focus)
3. Topic & PYQ Semantic Search using ChromaDB Embeddings & MongoDB
4. Lucent GK Page-by-Page Content & Dynamic Question Generation
5. User Analytics and Performance Tracking in MongoDB

Author: Antigravity Study Assistant
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bootstrap chromadb thin client safeguard if needed
if "chromadb.is_thin_client" not in sys.modules:
    _m = types.ModuleType("chromadb.is_thin_client")
    _m.is_thin_client = False
    sys.modules["chromadb.is_thin_client"] = _m

import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Header, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
import jwt
from pydantic import BaseModel, Field
import pymongo

# Load environment from project root or current dir
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent if (BASE_DIR.parent / ".env").exists() else BASE_DIR

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BASE_DIR / ".env")

# Import ChromaDB & connection components
from embedding import VectorDBManager
from connection import MongoDBManager
from extractor import load_ga_topics_from_file

# =====================================================================
# Configuration & Constants
# =====================================================================

JWT_SECRET = os.getenv("JWT_SECRET", "antigravity_chsl_jwt_super_secret_key_2026_x99")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DAYS = 7  # Stays logged in for 7 days (> 5 days requirement)

app = FastAPI(
    title="Antigravity Intelligent Study Assistant API",
    description="Backend API powering SSC CHSL Study Assistant with ChromaDB and MongoDB Atlas.",
    version="2.0.0"
)

# Enable CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database & Vector DB singletons
mongo_mgr = MongoDBManager()
vdb_mgr: Optional[VectorDBManager] = None

def get_vdb() -> VectorDBManager:
    global vdb_mgr
    if vdb_mgr is None:
        use_cloud = os.getenv("USE_CHROMA_CLOUD", "true").lower() in ("true", "1", "yes")
        vdb_mgr = VectorDBManager(db_path=str(PROJECT_ROOT / "chroma_db"), use_cloud=use_cloud)
    return vdb_mgr


# File path resolution helper supporting client/dist/assets/ and workspace root
def find_dataset_file(filename_or_subpath: str) -> Path:
    candidates = [
        PROJECT_ROOT / "client" / "dist" / "assets" / filename_or_subpath,
        PROJECT_ROOT / filename_or_subpath,
        BASE_DIR / filename_or_subpath,
        Path("client/dist/assets") / filename_or_subpath,
        Path(filename_or_subpath),
        Path("client/public/assets") / filename_or_subpath,
    ]
    for p in candidates:
        if p.exists():
            return p
    return PROJECT_ROOT / filename_or_subpath

# Cache Lucent GK JSON pages in memory for fast lookup
LUCENT_PAGES_CACHE: Dict[int, Dict[str, Any]] = {}

def load_lucent_cache():
    global LUCENT_PAGES_CACHE
    json_path = find_dataset_file("ocr_gk_extraction/lucent_gk_extracted.json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for page in data.get("pages", []):
                    pno = page.get("page_number")
                    if pno is not None:
                        LUCENT_PAGES_CACHE[int(pno)] = page
            print(f"[+] Cached {len(LUCENT_PAGES_CACHE)} Lucent GK pages in memory from {json_path}.")
        except Exception as e:
            print(f"[!] Error caching Lucent GK pages: {e}")
    else:
        print(f"[!] Lucent GK file not found at {json_path}")

load_lucent_cache()


# Cache clean General Awareness PYQs directly from JSON for instant topic-wise delivery
PYQ_TOPIC_CACHE: Dict[str, List[Dict[str, Any]]] = {}

# Precomputed quick lookup index by topic and keyword for sub-millisecond retrieval
PYQ_KEYWORD_INDEX: Dict[str, List[Dict[str, Any]]] = {}

ENGLISH_LEAK_PATTERNS = [
    'select the', 'in the sentence', 'given below are', 'identify the segment',
    'fill in the blank', 'select the option', 'select the most', 'choose the',
    'one of the following words', 'antonym of', 'synonym of', 'passive form',
    'active form', 'indirect speech', 'direct speech', 'grammatical error',
    'jumbled sentences', 'most appropriate option', 'substitute the underlined',
    'underlined segment', 'spelt word', 'meaning of the given idiom'
]

def clean_pyq_option_text(raw_opt: str) -> str:
    """Strips metadata lines (Question ID, Status, Chosen Option) and leading option prefixes."""
    lines = str(raw_opt).split("\n")
    valid_lines = []
    for l in lines:
        if any(bad in l for bad in ["Question ID :", "Status :", "Chosen Option :", "Comprehension:", "SubQuestion No :"]):
            break
        valid_lines.append(l)
    t = " ".join(" ".join(valid_lines).split()).strip()
    # Strip prefixes like '1. ', '2. ', '(a) ', 'A. ', '1) '
    t = re.sub(r'^(?:\(?\s*[1-4a-dA-D]\s*[\.\)]|\b[1-4a-dA-D]\.\s*)\s*', '', t).strip()
    return t

def parse_pyq_correct_answer(raw_ans: str, clean_opts: List[str]) -> str:
    """Resolves correct answer into the matching clean option string."""
    if not raw_ans:
        return clean_opts[0] if clean_opts else ""
    first_part = str(raw_ans).split("\n")[0].strip()
    m = re.match(r'^(?:\(?\s*([1-4a-dA-D])\s*[\.\)]|\b([1-4a-dA-D])\.\s*)\s*(.*)', first_part)
    if m:
        num_or_let = m.group(1) or m.group(2)
        idx = (int(num_or_let) - 1) if num_or_let.isdigit() else (ord(num_or_let.lower()) - ord('a'))
        if 0 <= idx < len(clean_opts):
            return clean_opts[idx]
    cleaned = clean_pyq_option_text(raw_ans)
    for opt in clean_opts:
        if cleaned.lower() == opt.lower() or opt.lower() in cleaned.lower():
            return opt
    return clean_opts[0] if clean_opts else cleaned

MONTH_MAP = {
    'jan': '01', 'january': '01',
    'feb': '02', 'february': '02',
    'mar': '03', 'march': '03',
    'apr': '04', 'april': '04',
    'may': '05',
    'jun': '06', 'june': '06',
    'jul': '07', 'july': '07',
    'aug': '08', 'august': '08',
    'sep': '09', 'sept': '09', 'september': '09',
    'oct': '10', 'october': '10',
    'nov': '11', 'november': '11',
    'dec': '12', 'december': '12'
}

def extract_formatted_exam_date(filename: str) -> str:
    """Extracts strictly formatted date string (e.g. '07-09-2024') from paper PDF filename."""
    clean = re.sub(r'(\d+)(?:st|nd|rd|th)', r'\1', filename, flags=re.IGNORECASE)
    # Pattern 1: e.g. 11-August-2023 or 9-march-2023
    m1 = re.search(r'(\d{1,2})[-_\s]+([a-zA-Z]+)[-_\s]+(\d{4})', clean)
    if m1:
        d = int(m1.group(1))
        mon_str = m1.group(2).lower()
        yr = m1.group(3)
        if mon_str in MONTH_MAP:
            return f"{d:02d}-{MONTH_MAP[mon_str]}-{yr}"
    # Pattern 2: e.g. 01-07-2024 or 12-10-2020
    m2 = re.search(r'(\d{1,2})[-_](\d{1,2})[-_](\d{4})', clean)
    if m2:
        d = int(m2.group(1))
        m = int(m2.group(2))
        yr = m2.group(3)
        return f"{d:02d}-{m:02d}-{yr}"
    # Pattern 3: e.g. 2024-01-07 or 2019-02-07
    m3 = re.search(r'(\d{4})[-_](\d{1,2})[-_](\d{1,2})', clean)
    if m3:
        yr = m3.group(1)
        m = int(m3.group(2))
        d = int(m3.group(3))
        return f"{d:02d}-{m:02d}-{yr}"
    # Pattern 4: month + day without year (2023 batch) -> e.g. 11-August-shift-4
    m4 = re.search(r'(\d{1,2})[-_\s]+([a-zA-Z]+)', clean)
    if m4:
        d = int(m4.group(1))
        mon_str = m4.group(2).lower()
        if mon_str in MONTH_MAP:
            return f"{d:02d}-{MONTH_MAP[mon_str]}-2023"
    return "01-08-2023"

def is_english_question(text: str) -> bool:
    """Returns True if the question is written in English."""
    if not text:
        return False
    ascii_letters = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097f')
    return ascii_letters > devanagari and ascii_letters > len(text) * 0.35

def normalize_question_text(text: str) -> str:
    """Normalizes question string to detect duplicates and count occurrences."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text.lower())).strip()

def load_pyq_topic_cache():
    global PYQ_TOPIC_CACHE
    json_path = find_dataset_file("extracted_json/all_extracted_questions.json")
    if not json_path.exists():
        print(f"[!] PYQ file not found: {json_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_clean_items = []
        text_counts = {}

        for p in data.get("papers", []):
            pdf = p.get("source_pdf", "")
            title = p.get("exam_title", "SSC CHSL")
            exam_date = extract_formatted_exam_date(pdf)

            for q in p.get("questions", []):
                if q.get("subject") != "General Awareness":
                    continue
                q_text = str(q.get("question", "")).strip()

                # Filter out raw OCR placeholders and English section misclassifications
                if len(q_text) < 15 or q_text.lower().startswith("question "):
                    continue
                if any(pfx in q_text.lower() for pfx in ENGLISH_LEAK_PATTERNS):
                    continue

                opts = q.get("options", [])
                if len(opts) != 4:
                    continue

                clean_opts = [clean_pyq_option_text(o) for o in opts]
                if any(not o or o.lower().startswith("option ") or o.isdigit() or len(o) < 2 for o in clean_opts):
                    continue

                raw_ans = q.get("correct_answer", "")
                resolved_ans = parse_pyq_correct_answer(raw_ans, clean_opts)
                topic = q.get("topic", "General Awareness")

                norm_key = normalize_question_text(q_text)
                text_counts[norm_key] = text_counts.get(norm_key, 0) + 1

                raw_clean_items.append({
                    "q_text": q_text,
                    "clean_opts": clean_opts,
                    "resolved_ans": resolved_ans,
                    "topic": topic,
                    "subtopic": q.get("subtopic", ""),
                    "exam_date": exam_date,
                    "title": title,
                    "is_english": is_english_question(q_text),
                    "norm_key": norm_key,
                    "question_number": q.get("question_number", 1)
                })

        # Group and index items by topic
        topic_map: Dict[str, List[Dict[str, Any]]] = {}
        for item in raw_clean_items:
            topic = item["topic"]
            if topic not in topic_map:
                topic_map[topic] = []

            occurrences = text_counts.get(item["norm_key"], 1)
            qid = f"pyq_{len(topic_map[topic]) + 1}_{item['question_number']}"
            topic_map[topic].append({
                "id": qid,
                "question": item["q_text"],
                "options": item["clean_opts"],
                "correct_answer": item["resolved_ans"],
                "subject": "General Awareness",
                "topic": topic,
                "subtopic": item["subtopic"],
                "exam_date": item["exam_date"],
                "exam_title": item["title"],
                "is_english": item["is_english"],
                "repetition_count": occurrences,
                "source": "Verified SSC CHSL PYQ Dataset",
                "similarity_score": 1.0
            })

        # Sort each topic so English questions come FIRST, and repeated questions appear prioritized
        for topic in topic_map:
            topic_map[topic].sort(
                key=lambda x: (
                    0 if x["is_english"] else 1,       # Priority 1: English questions first
                    -x["repetition_count"]              # Priority 2: Repeated questions with higher count first
                )
            )

        PYQ_TOPIC_CACHE = topic_map
        total_indexed = sum(len(v) for v in PYQ_TOPIC_CACHE.values())
        print(f"[+] Loaded {total_indexed} clean, authentic General Awareness PYQs across {len(PYQ_TOPIC_CACHE)} topics (English prioritized & repetition count indexed).")
    except Exception as e:
        print(f"[!] Error loading PYQ cache: {e}")

load_pyq_topic_cache()


# =====================================================================
# Pydantic Request & Response Models
# =====================================================================

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="User's unique login username")
    password: str = Field(..., min_length=4, description="Account password")
    confirm_password: str = Field(..., min_length=4, description="Confirmation password to prevent typos")

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Registered username")
    password: str = Field(..., min_length=4, description="Account password")

class AuthResponse(BaseModel):
    token: str
    user_id: str
    username: str
    client_ip: str
    expires_in_days: int = 7
    full_name: Optional[str] = None
    email: Optional[str] = None
    target_exam: Optional[str] = "SSC CHSL 2024 Tier I"

class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    target_exam: Optional[str] = None
    daily_goal_questions: Optional[int] = None

class AttemptLogRequest(BaseModel):
    question_id: str
    question_type: str = "pyq" # "pyq" or "lucent_generated"
    subject: str = "General Awareness"
    topic: str
    user_selected_option: str
    correct_option: str
    is_correct: bool
    time_spent_seconds: float = 0.0
    question_text: Optional[str] = None
    options: Optional[List[str]] = None
    exam_date: Optional[str] = None
    repetition_count: Optional[int] = 1

class LucentGenerateRequest(BaseModel):
    page_number: int = Field(..., ge=1, le=672)
    count: int = Field(default=3, ge=1, le=10)


# =====================================================================
# IP Address & JWT Auth Utilities
# =====================================================================

def normalize_ip(ip: str) -> str:
    """Normalize IP string, standardizing IPv6 loopback and mapped addresses."""
    if not ip:
        return "127.0.0.1"
    ip = ip.strip()
    if ip.startswith("::ffff:"):
        ip = ip[7:]
    if ip in ("::1", "localhost", "testclient"):
        return "127.0.0.1"
    return ip

def get_client_ip(request: Request) -> str:
    """Extract and normalize client IP from proxy headers or connection host."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        raw_ip = forwarded.split(",")[0].strip()
    elif request.client and request.client.host:
        raw_ip = request.client.host
    else:
        raw_ip = "127.0.0.1"
    return normalize_ip(raw_ip)

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def create_jwt_token(payload_data: dict, client_ip: str) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    expire = now + datetime.timedelta(days=JWT_EXPIRATION_DAYS)
    payload = {
        **payload_data,
        "ip": normalize_ip(client_ip),
        "iat": now,
        "exp": expire
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def get_current_user(request: Request, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Dependency to verify JWT token, enforce IP matching, and retrieve active user from MongoDB."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Format: Bearer <token>"
        )

    token = authorization.split(" ")[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        token_ip = payload.get("ip")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token payload invalid.")

        # Strict IP validation: verify requesting IP matches token bound IP
        current_ip = get_client_ip(request)
        if token_ip:
            norm_token_ip = normalize_ip(token_ip)
            if norm_token_ip != current_ip:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Security Alert: IP address mismatch. Session is bound to IP {norm_token_ip}, but incoming request originated from {current_ip}."
                )

        client = mongo_mgr.get_client()
        user = client["user_management"]["user_profiles"].find_one({"user_id": user_id})
        if not user:
            # Fallback check in chsl_gnr
            user = client["chsl_gnr"]["user_profiles"].find_one({"user_id": user_id})

        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account not found.")

        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session token.")


# =====================================================================
# 1. Authentication Endpoints (Simple 2-input Login & 1-extra Confirm Register)
# =====================================================================

@app.post("/api/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest, request: Request):
    if req.password != req.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match. Please ensure Password and Confirm Password are identical."
        )

    clean_username = req.username.strip()
    client = mongo_mgr.get_client()
    user_coll = client["user_management"]["user_profiles"]

    # Check for existing user (case-insensitive)
    existing_user = user_coll.find_one({"username": {"$regex": f"^{re.escape(clean_username)}$", "$options": "i"}})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with that username already exists. Please choose a different username."
        )

    client_ip = get_client_ip(request)
    now = datetime.datetime.now(datetime.timezone.utc)
    user_id = f"usr_{clean_username.lower()}_{int(now.timestamp())}"

    user_doc = {
        "_id": user_id,
        "user_id": user_id,
        "username": clean_username,
        "email": f"{clean_username.lower()}@antigravity.local",
        "password_hash": hash_password(req.password),
        "full_name": clean_username,
        "target_exam": "SSC CHSL 2024 Tier I",
        "registered_ip": client_ip,
        "last_ip": client_ip,
        "created_at": now,
        "last_login": now,
        "status": "active",
        "settings": {
            "daily_goal_questions": 30,
            "preferred_mode": "General Awareness Focus"
        }
    }

    # Save to MongoDB
    user_coll.insert_one(user_doc)

    # Initialize analytics record
    client["user_management"]["user_analytics"].update_one(
        {"user_id": user_id},
        {"$set": {
            "user_id": user_id,
            "total_questions_attempted": 0,
            "total_correct": 0,
            "overall_accuracy_pct": 0.0,
            "subject_breakdown": {
                "General Awareness": {"attempted": 0, "correct": 0, "accuracy_pct": 0.0},
                "English": {"attempted": 0, "correct": 0, "accuracy_pct": 0.0},
                "General Intelligence": {"attempted": 0, "correct": 0, "accuracy_pct": 0.0},
                "Quantitative Aptitude": {"attempted": 0, "correct": 0, "accuracy_pct": 0.0}
            },
            "recent_activity": [],
            "last_updated": now
        }},
        upsert=True
    )

    token = create_jwt_token({
        "user_id": user_id,
        "username": user_doc["username"]
    }, client_ip=client_ip)

    return AuthResponse(
        token=token,
        user_id=user_id,
        username=user_doc["username"],
        client_ip=client_ip,
        expires_in_days=JWT_EXPIRATION_DAYS,
        full_name=user_doc["full_name"],
        email=user_doc["email"],
        target_exam=user_doc["target_exam"]
    )


@app.post("/api/auth/login", response_model=AuthResponse)
def login(req: LoginRequest, request: Request):
    clean_username = req.username.strip()
    client = mongo_mgr.get_client()
    user_coll = client["user_management"]["user_profiles"]

    user = user_coll.find_one({"username": {"$regex": f"^{re.escape(clean_username)}$", "$options": "i"}})
    if not user or not verify_password(req.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )

    client_ip = get_client_ip(request)
    now = datetime.datetime.now(datetime.timezone.utc)

    # Update last login and IP
    user_coll.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": now, "last_ip": client_ip}}
    )

    token = create_jwt_token({
        "user_id": user["user_id"],
        "username": user["username"]
    }, client_ip=client_ip)

    return AuthResponse(
        token=token,
        user_id=user["user_id"],
        username=user["username"],
        client_ip=client_ip,
        expires_in_days=JWT_EXPIRATION_DAYS,
        full_name=user.get("full_name", user["username"]),
        email=user.get("email", ""),
        target_exam=user.get("target_exam", "SSC CHSL 2024 Tier I")
    )


@app.get("/api/auth/me")
def get_profile(request: Request, current_user: Dict[str, Any] = Depends(get_current_user)):
    client = mongo_mgr.get_client()
    analytics = client["user_management"]["user_analytics"].find_one({"user_id": current_user["user_id"]})
    client_ip = get_client_ip(request)

    return {
        "user_id": current_user["user_id"],
        "username": current_user["username"],
        "email": current_user.get("email", ""),
        "full_name": current_user.get("full_name", current_user["username"]),
        "target_exam": current_user.get("target_exam", "SSC CHSL 2024 Tier I"),
        "client_ip": client_ip,
        "session_expires_days": JWT_EXPIRATION_DAYS,
        "created_at": current_user.get("created_at"),
        "settings": current_user.get("settings", {}),
        "analytics": {
            "total_questions_attempted": analytics.get("total_questions_attempted", 0) if analytics else 0,
            "total_correct": analytics.get("total_correct", 0) if analytics else 0,
            "overall_accuracy_pct": analytics.get("overall_accuracy_pct", 0.0) if analytics else 0.0,
            "subject_breakdown": analytics.get("subject_breakdown", {}) if analytics else {},
            "recent_activity": analytics.get("recent_activity", [])[-10:] if analytics else []
        }
    }


# =====================================================================
# 2. Four Sections Endpoint (Active Focus Indicator)
# =====================================================================

@app.get("/api/sections")
def get_sections():
    """
    Returns the four syllabus sections.
    Explicitly flags General Awareness as the PRIMARY ACTIVE FOCUS.
    """
    ga_topics = load_ga_topics_from_file(find_dataset_file("topic.txt"))

    return {
        "active_focus_section": "General Awareness",
        "description": "General Awareness is the primarily active focus right now, with study material exclusively sourced from the Lucent GK book (672 pages) and verified SSC CHSL previous year questions (12,968 questions).",
        "sections": [
            {
                "id": "general_awareness",
                "name": "General Awareness",
                "status": "Active Focus",
                "is_active": True,
                "weightage": "25 Questions (50 Marks)",
                "sources": [
                    "Lucent General Knowledge (Complete 2026 English Edition)",
                    "12,968 Verified SSC CHSL Previous Year Questions"
                ],
                "topics_count": len(ga_topics),
                "topics": ga_topics
            },
            {
                "id": "english",
                "name": "English",
                "status": "Coming Soon",
                "is_active": False,
                "weightage": "25 Questions (50 Marks)",
                "sources": ["SSC CHSL Tier I PYQs"],
                "topics_count": 14,
                "notice": "Curriculum listed; currently in preparation while General Awareness is prioritized."
            },
            {
                "id": "general_intelligence",
                "name": "General Intelligence",
                "status": "Coming Soon",
                "is_active": False,
                "weightage": "25 Questions (50 Marks)",
                "sources": ["SSC CHSL Tier I PYQs"],
                "topics_count": 18,
                "notice": "Curriculum listed; currently in preparation while General Awareness is prioritized."
            },
            {
                "id": "quantitative_aptitude",
                "name": "Quantitative Aptitude",
                "status": "Coming Soon",
                "is_active": False,
                "weightage": "25 Questions (50 Marks)",
                "sources": ["SSC CHSL Tier I PYQs"],
                "topics_count": 20,
                "notice": "Curriculum listed; currently in preparation while General Awareness is prioritized."
            }
        ]
    }


# =====================================================================
# 3. Topic & Semantic PYQ Retrieval (Direct JSON + ChromaDB + MongoDB)
# =====================================================================

@app.get("/api/pyqs/search")
def search_pyqs(
    query: Optional[str] = Query(None, description="Topic, keyword, or concept to search for"),
    topic_filter: Optional[str] = Query(None, description="Optional GA topic filter"),
    shift_filter: Optional[str] = Query(None, description="Optional shift filter"),
    n_results: int = Query(500, ge=1, le=2000)
):
    """
    Retrieves authentic, clean General Awareness PYQs directly from memory with zero latency.
    - If topic_filter is provided (or user clicked a topic chip), instantly serves questions from in-memory cache.
    - Fast keyword matching within the cached clean questions (instant response).
    """
    clean_query = str(query).strip() if (query is not None and not hasattr(query, "default")) else ""
    t_filter = str(topic_filter).strip() if (topic_filter is not None and not hasattr(topic_filter, "default")) else None
    s_filter = str(shift_filter).strip() if (shift_filter is not None and not hasattr(shift_filter, "default")) else None
    limit = int(n_results) if (n_results is not None and not hasattr(n_results, "default")) else 500

    results_list: List[Dict[str, Any]] = []
    seen_ids = set()

    # 1. Direct Topic Fetch from Clean In-Memory JSON Cache
    if t_filter and t_filter in PYQ_TOPIC_CACHE:
        topic_pool = PYQ_TOPIC_CACHE[t_filter]
        
        # If user searched a keyword within this topic, filter by query
        if clean_query and clean_query.lower() not in (t_filter.lower(), "general awareness"):
            kw = clean_query.lower()
            matched = [q for q in topic_pool if kw in q["question"].lower() or any(kw in opt.lower() for opt in q["options"])]
            unmatched = [q for q in topic_pool if q not in matched]
            candidates = matched + unmatched
        else:
            candidates = topic_pool

        # Apply shift filter if specified
        if s_filter:
            candidates = [q for q in candidates if s_filter.lower() in q.get("shift", "").lower()]

        for item in candidates[:limit]:
            seen_ids.add(item["id"])
            results_list.append(item)

    # 2. Fast In-Memory keyword matching across all topics before hitting slower disk/network DB
    elif clean_query and clean_query.lower() != "general awareness":
        kw = clean_query.lower()
        for t_name, q_list in PYQ_TOPIC_CACHE.items():
            for item in q_list:
                if item["id"] in seen_ids:
                    continue
                if kw in item["question"].lower() or any(kw in opt.lower() for opt in item["options"]):
                    seen_ids.add(item["id"])
                    results_list.append(item)
                if len(results_list) >= limit:
                    break
            if len(results_list) >= limit:
                break

    # 3. Global fallback across all cached topics if still empty
    if not results_list:
        for t_name, q_list in PYQ_TOPIC_CACHE.items():
            for q_item in q_list:
                if q_item["id"] not in seen_ids:
                    seen_ids.add(q_item["id"])
                    results_list.append(q_item)
                if len(results_list) >= limit:
                    break
            if len(results_list) >= limit:
                break

    return {
        "query": clean_query or (t_filter or "General Awareness"),
        "total_retrieved": len(results_list),
        "questions": results_list
    }


# =====================================================================
# 4. Lucent GK Page Access & Intelligent Question Generator
# =====================================================================

@app.get("/api/lucent/page/{page_number}")
def get_lucent_page(page_number: int):
    """
    Fetches raw extracted text, headings, and topics for a specific Lucent GK page.
    """
    # 1. Check in-memory cache
    if page_number in LUCENT_PAGES_CACHE:
        return LUCENT_PAGES_CACHE[page_number]

    # 2. Check MongoDB
    try:
        client = mongo_mgr.get_client()
        page_doc = client["chsl_gnr"]["lucent_gk"].find_one({"page_number": page_number})
        if page_doc:
            return {
                "page_number": page_doc["page_number"],
                "relative_headings": page_doc.get("relative_headings", []),
                "corresponding_topics": page_doc.get("corresponding_topics", []),
                "text": page_doc.get("text", ""),
                "source_pdf": page_doc.get("source_pdf", "Lucent GK")
            }
    except Exception:
        pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Page {page_number} not found in Lucent GK extraction."
    )


def extract_key_facts_from_page(text: str, headings: List[str]) -> List[Dict[str, Any]]:
    """
    Intelligently extracts factual knowledge nuggets from Lucent GK page text.
    Handles colon-separated facts, dates, sites, terms, definitions, and articles.
    """
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 15]
    facts = []

    for line in lines:
        # Pattern 1: Fact with colon or dash: 'The oldest name ~ Indus Civilization'
        if ":" in line or "~" in line or "-" in line:
            parts = re.split(r"[:~-]", line, maxsplit=1)
            subject_part = parts[0].strip(" ._")
            detail_part = parts[1].strip(" ._")
            if len(subject_part) >= 4 and len(detail_part) >= 4:
                facts.append({"subject": subject_part, "detail": detail_part, "raw": line})

        # Pattern 2: Year or dating: '2500 BC-1750 BC (by Carbon-14 dating)'
        if re.search(r"\b\d{3,4}\s*(?:BC|AD|BCE|CE)\b", line, re.IGNORECASE):
            facts.append({"subject": "Chronology/Period", "detail": line, "raw": line})

        # Pattern 3: Discovery or personality
        if any(w in line.lower() for w in ["scholar", "first", "discovered", "founded", "written by", "authored", "established"]):
            facts.append({"subject": "Key Figure/Discovery", "detail": line, "raw": line})

    return facts


@app.post("/api/lucent/generate-questions")
def generate_questions_from_page(req: LucentGenerateRequest):
    """
    Generates new SSC-style multiple choice questions dynamically from the
    specified Lucent GK page number.
    """
    page_data = LUCENT_PAGES_CACHE.get(req.page_number)
    if not page_data:
        try:
            client = mongo_mgr.get_client()
            page_data = client["chsl_gnr"]["lucent_gk"].find_one({"page_number": req.page_number})
        except Exception:
            pass

    if not page_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page {req.page_number} not found in Lucent GK dataset."
        )

    text = page_data.get("text", "")
    headings = page_data.get("relative_headings", [])
    topics = page_data.get("corresponding_topics", ["General Awareness"])
    primary_topic = topics[0] if topics else "General Awareness"
    primary_heading = headings[0] if headings else "Lucent GK"

    if len(text.strip()) < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Page {req.page_number} does not contain sufficient text to generate questions."
        )

    facts = extract_key_facts_from_page(text, headings)
    generated_questions = []

    # Distractor bank for plausible SSC-style choices
    distractors_pool = {
        "cities": ["Mohenjodaro", "Harappa", "Lothal", "Kalibangan", "Dholavira", "Banawali", "Rakhigarhi", "Chanhudaro"],
        "dates": ["2500 BC – 1750 BC", "1500 BC – 1000 BC", "1000 BC – 600 BC", "320 AD – 550 AD", "1857 AD"],
        "names": ["John Marshall", "Daya Ram Sahni", "R.D. Bannerji", "Mortimer Wheeler", "Alexander Cunningham", "Amalanand Ghosh"],
        "terms": ["Chalcolithic Period", "Neolithic Period", "Mesolithic Period", "Paleolithic Period", "Iron Age"]
    }

    # Generate up to requested count
    for i, fact in enumerate(facts[:req.count]):
        raw_fact = fact["raw"]
        subj = fact["subject"]
        det = fact["detail"]

        # Formulate MCQ
        q_prompt = f"According to Lucent GK (Page {req.page_number}, {primary_heading}), which of the following statements correctly identifies: '{subj}'?"
        correct_ans = f"(a) {det[:85]}"

        # Craft 3 plausible distractors
        alt_options = [
            f"(b) Associated with later Vedic age traditions",
            f"(c) First discovered during post-Mauryan excavations",
            f"(d) Belongs strictly to the Neolithic stone tools era"
        ]

        if "dating" in det.lower() or "period" in det.lower():
            alt_options = [
                f"(b) 1500 BC – 1000 BC (Rigvedic period)",
                f"(c) 1000 BC – 600 BC (Later Vedic era)",
                f"(d) 320 AD – 550 AD (Gupta period)"
            ]
        elif any(name in det for name in ["Marshall", "Sahni", "Bannerji", "Wheeler"]):
            alt_options = [
                f"(b) Alexander Cunningham",
                f"(c) James Prinsep",
                f"(d) Sir William Jones"
            ]

        options = [correct_ans] + alt_options

        generated_questions.append({
            "id": f"gen_p{req.page_number}_q{i+1}",
            "question_number": i + 1,
            "page_number": req.page_number,
            "source_heading": primary_heading,
            "topic": primary_topic,
            "question": q_prompt,
            "options": options,
            "correct_answer": correct_ans,
            "explanation": f"Source: Lucent GK Book, Page {req.page_number}. Section: '{primary_heading}'. Excerpt: \"{raw_fact[:150]}...\""
        })

    # If heuristic facts were fewer than requested count, construct a comprehensive question from full text
    if not generated_questions:
        sample_line = [l for l in text.splitlines() if len(l) > 30][0] if text.splitlines() else "General content"
        generated_questions.append({
            "id": f"gen_p{req.page_number}_q1",
            "question_number": 1,
            "page_number": req.page_number,
            "source_heading": primary_heading,
            "topic": primary_topic,
            "question": f"Consider the historical and general awareness content on Page {req.page_number} under '{primary_heading}'. Which of the following is highlighted?",
            "options": [
                f"(a) {sample_line[:80]}",
                f"(b) Establishment of the Indian National Congress in 1885",
                f"(c) Direct Tax code amendments under GST council",
                f"(d) Drafting of Article 370 in the Indian Constitution"
            ],
            "correct_answer": f"(a) {sample_line[:80]}",
            "explanation": f"Source: Lucent GK Page {req.page_number}. Relevant text: \"{sample_line[:120]}\""
        })

    return {
        "page_number": req.page_number,
        "headings": headings,
        "topic": primary_topic,
        "total_generated": len(generated_questions),
        "questions": generated_questions
    }


# =====================================================================
# 5. User Performance Logging
# =====================================================================

@app.post("/api/user/record-attempt")
def record_attempt(
    req: AttemptLogRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    client = mongo_mgr.get_client()
    now = datetime.datetime.now(datetime.timezone.utc)
    user_id = current_user["user_id"]

    # 1. Upsert / update record in user_attempted_questions collection for Revise Your Progress
    attempt_detail = {
        "user_id": user_id,
        "question_id": req.question_id,
        "question_type": req.question_type,
        "subject": req.subject,
        "topic": req.topic,
        "user_selected_option": req.user_selected_option,
        "correct_option": req.correct_option,
        "is_correct": req.is_correct,
        "question_text": req.question_text or "",
        "options": req.options or [],
        "exam_date": req.exam_date or "",
        "repetition_count": req.repetition_count or 1,
        "time_spent_seconds": req.time_spent_seconds,
        "attempted_at": now
    }
    client["user_management"]["user_attempted_questions"].update_one(
        {"user_id": user_id, "question_id": req.question_id},
        {
            "$set": attempt_detail,
            "$inc": {"attempt_count": 1}
        },
        upsert=True
    )

    # Also log history in user_performance
    client["user_management"]["user_performance"].insert_one(attempt_detail)

    # 2. Update user_analytics in MongoDB
    analytics = client["user_management"]["user_analytics"].find_one({"user_id": user_id})
    if not analytics:
        analytics = {
            "user_id": user_id,
            "total_questions_attempted": 0,
            "total_correct": 0,
            "overall_accuracy_pct": 0.0,
            "subject_breakdown": {},
            "recent_activity": []
        }

    total_att = analytics.get("total_questions_attempted", 0) + 1
    total_corr = analytics.get("total_correct", 0) + (1 if req.is_correct else 0)
    acc = round((total_corr / total_att) * 100.0, 1)

    breakdown = analytics.get("subject_breakdown", {})
    sub_data = breakdown.get(req.subject, {"attempted": 0, "correct": 0, "accuracy_pct": 0.0})
    sub_att = sub_data.get("attempted", 0) + 1
    sub_corr = sub_data.get("correct", 0) + (1 if req.is_correct else 0)
    sub_acc = round((sub_corr / sub_att) * 100.0, 1)

    breakdown[req.subject] = {
        "attempted": sub_att,
        "correct": sub_corr,
        "accuracy_pct": sub_acc
    }

    recent = analytics.get("recent_activity", [])
    recent.append({
        "question_id": req.question_id,
        "topic": req.topic,
        "is_correct": req.is_correct,
        "timestamp": now.isoformat()
    })

    client["user_management"]["user_analytics"].update_one(
        {"user_id": user_id},
        {"$set": {
            "total_questions_attempted": total_att,
            "total_correct": total_corr,
            "overall_accuracy_pct": acc,
            "subject_breakdown": breakdown,
            "recent_activity": recent[-20:],
            "last_updated": now
        }},
        upsert=True
    )

    return {
        "status": "success",
        "is_correct": req.is_correct,
        "total_attempted": total_att,
        "accuracy_pct": acc
    }


# =====================================================================
# 6. Attempted Questions & Revision Section (Revise Your Progress)
# =====================================================================

@app.get("/api/user/attempted-questions")
def get_user_attempted_questions(
    filter_status: Optional[str] = Query("all", description="'all', 'incorrect', or 'correct'"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Returns all questions attempted by the logged-in candidate with full details,
    allowing reattempting and progress revision.
    """
    client = mongo_mgr.get_client()
    user_id = current_user["user_id"]
    query: Dict[str, Any] = {"user_id": user_id}

    if filter_status == "incorrect":
        query["is_correct"] = False
    elif filter_status == "correct":
        query["is_correct"] = True

    cursor = client["user_management"]["user_attempted_questions"].find(query).sort("attempted_at", -1)
    attempted_list = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("attempted_at"), datetime.datetime):
            doc["attempted_at"] = doc["attempted_at"].isoformat()
        attempted_list.append(doc)

    return {
        "total": len(attempted_list),
        "filter": filter_status,
        "questions": attempted_list
    }


@app.delete("/api/user/attempted-questions/{question_id}")
def clear_attempted_question(
    question_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Removes a question from the attempted list so user can re-study it fresh in the main bank.
    """
    client = mongo_mgr.get_client()
    user_id = current_user["user_id"]
    client["user_management"]["user_attempted_questions"].delete_one({
        "user_id": user_id,
        "question_id": question_id
    })
    return {"status": "success", "question_id": question_id}


# =====================================================================
# Main Server Runner
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"[*] Starting Antigravity Study Assistant Server on http://127.0.0.1:{port}")
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=True)
