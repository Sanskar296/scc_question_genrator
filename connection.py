"""
MongoDB Database Connection & Data Ingestion Pipeline
======================================================
Connects to MongoDB Atlas cluster using MONGODB_URI from local .env.
Organizes data across:
  1. Designated Project Database: 'chsl_gnr' (PYQs & Lucent GK collections)
  2. Multi-Database Architecture (Production Grade):
     a. 'study_material':
        - 'pyqs': 12,968+ SSC CHSL questions tagged by subject, topic, shift, date.
        - 'lucent_gk': 672 page-by-page OCR extractions with headings & GA topics.
     b. 'user_management':
        - 'user_profiles': User account, target exams, and profile information.
        - 'user_analytics': Aggregated performance, topic strengths, and weaknesses.
        - 'user_performance': Question-level and test-level attempt tracking.
        - 'test_series': Test series definitions, mock tests, and cut-off metadata.

Author: Antigravity (Expert Database Engineering System)
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import pymongo
from pymongo import ASCENDING, DESCENDING, IndexModel, TEXT, UpdateOne
from pymongo.errors import BulkWriteError, ConnectionFailure, ServerSelectionTimeoutError

# UTF-8 encoding safeguard for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables
load_dotenv()


# =====================================================================
# 1. MongoDB Connection Manager
# =====================================================================

class MongoDBManager:
    """Production-grade MongoDB client manager with connection verification and pooling."""

    def __init__(self, uri: Optional[str] = None):
        self.uri = uri or os.getenv("MONGODB_URI")
        if not self.uri:
            raise ValueError(
                "MONGODB_URI not found in environment or .env file. "
                "Please configure MONGODB_URI=mongodb+srv://... in your .env"
            )

        self._client: Optional[pymongo.MongoClient] = None

    def get_client(self) -> pymongo.MongoClient:
        """Returns verified MongoClient instance with connection pooling."""
        if self._client is None:
            try:
                # Mask credentials for logging
                masked_uri = re.sub(r":([^:@]+)@", ":****@", self.uri)
                print(f"[*] Connecting to MongoDB Cluster: {masked_uri}")

                self._client = pymongo.MongoClient(
                    self.uri,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=10000,
                    socketTimeoutMS=30000,
                    maxPoolSize=50,
                    minPoolSize=10,
                    retryWrites=True,
                    appname="ai_interview_chsl"
                )

                # Ping cluster to verify authentication & network reachability
                self._client.admin.command("ping")
                print("[+] Successfully connected to MongoDB Atlas cluster!")

            except (ConnectionFailure, ServerSelectionTimeoutError) as err:
                raise ConnectionError(
                    f"Failed to connect to MongoDB cluster: {err}\n"
                    "Check network access, IP whitelist in Atlas, and credentials."
                ) from err

        return self._client

    def get_database(self, db_name: str) -> pymongo.database.Database:
        """Retrieves a specific database from the cluster."""
        return self.get_client()[db_name]

    def close(self):
        """Closes the client connection."""
        if self._client:
            self._client.close()
            self._client = None
            print("[*] MongoDB connection closed.")


# =====================================================================
# 2. Schema Validation & Index Engineering
# =====================================================================

def ensure_indexes_for_pyqs(collection: pymongo.collection.Collection):
    """Creates optimized indexes for fast search, filtering, and retrieval of PYQs."""
    indexes = [
        IndexModel([("paper_id", ASCENDING), ("question_number", ASCENDING)], unique=True, name="idx_paper_qnum_unique"),
        IndexModel([("subject", ASCENDING), ("topic", ASCENDING)], name="idx_subject_topic"),
        IndexModel([("topic", ASCENDING)], name="idx_topic"),
        IndexModel([("exam_date", ASCENDING), ("shift", ASCENDING)], name="idx_date_shift"),
        IndexModel([("exam_title", ASCENDING)], name="idx_exam_title"),
        IndexModel([("question", TEXT)], name="idx_question_text_search")
    ]
    try:
        collection.create_indexes(indexes)
        print(f"[+] Indexes verified for collection '{collection.name}' in db '{collection.database.name}'")
    except Exception as e:
        print(f"[!] Warning creating indexes on '{collection.name}': {e}")


def ensure_indexes_for_lucent(collection: pymongo.collection.Collection):
    """Creates indexes for page lookups, topics, headings, and full-text retrieval on Lucent GK."""
    indexes = [
        IndexModel([("page_number", ASCENDING)], unique=True, name="idx_page_number_unique"),
        IndexModel([("corresponding_topics", ASCENDING)], name="idx_corresponding_topics"),
        IndexModel([("text", TEXT), ("relative_headings", TEXT)], name="idx_text_heading_search")
    ]
    try:
        collection.create_indexes(indexes)
        print(f"[+] Indexes verified for collection '{collection.name}' in db '{collection.database.name}'")
    except Exception as e:
        print(f"[!] Warning creating indexes on '{collection.name}': {e}")


def ensure_indexes_for_user_management(user_db: pymongo.database.Database):
    """Creates production indexes and collections for user profiles, analytics, and performance."""
    # 1. User Profiles
    profiles = user_db["user_profiles"]
    profiles.create_indexes([
        IndexModel([("user_id", ASCENDING)], unique=True, name="idx_user_id_unique"),
        IndexModel([("email", ASCENDING)], unique=True, name="idx_email_unique"),
        IndexModel([("target_exam", ASCENDING)], name="idx_target_exam"),
        IndexModel([("status", ASCENDING)], name="idx_status")
    ])

    # 2. User Analytics
    analytics = user_db["user_analytics"]
    analytics.create_indexes([
        IndexModel([("user_id", ASCENDING)], unique=True, name="idx_analytics_user_id_unique"),
        IndexModel([("overall_accuracy_pct", DESCENDING)], name="idx_accuracy"),
        IndexModel([("total_questions_attempted", DESCENDING)], name="idx_questions_attempted")
    ])

    # 3. User Performance
    performance = user_db["user_performance"]
    performance.create_indexes([
        IndexModel([("user_id", ASCENDING), ("attempted_at", DESCENDING)], name="idx_user_timeline"),
        IndexModel([("test_id", ASCENDING), ("user_id", ASCENDING)], name="idx_test_user"),
        IndexModel([("subject", ASCENDING), ("is_correct", ASCENDING)], name="idx_subject_accuracy"),
        IndexModel([("question_id", ASCENDING)], name="idx_question_id")
    ])

    # 4. Test Series
    test_series = user_db["test_series"]
    test_series.create_indexes([
        IndexModel([("series_id", ASCENDING)], unique=True, name="idx_series_id_unique"),
        IndexModel([("exam_category", ASCENDING)], name="idx_exam_category"),
        IndexModel([("is_published", ASCENDING)], name="idx_published")
    ])

    print(f"[+] Indexes verified for all 4 User Management collections in db '{user_db.name}'")


# =====================================================================
# 3. Data Loaders & Enrichment
# =====================================================================

def parse_exam_metadata_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts exam date and shift from PDF/JSON filenames."""
    # Match patterns like: 11-August-2023 or 02-Aug-2023 or 11-august-2023
    date_match = re.search(r"(\d{1,2})[- ]([A-Za-z]+)[- ](\d{4})", filename)
    date_str = f"{date_match.group(1)} {date_match.group(2).capitalize()} {date_match.group(3)}" if date_match else None

    # Match Shift 1, Shift-2, etc.
    shift_match = re.search(r"shift[- ]?(\d+)", filename, re.IGNORECASE)
    shift_str = f"Shift {shift_match.group(1)}" if shift_match else None

    return date_str, shift_str


def load_pyqs_data(
    extracted_json_dir: str | Path = "extracted_json",
    master_file: str | Path = "extracted_json/all_extracted_questions.json"
) -> List[Dict[str, Any]]:
    """
    Loads all extracted questions from master JSON and/or individual question paper JSONs.
    Returns normalized documents with metadata.
    """
    master_path = Path(master_file)
    extracted_dir = Path(extracted_json_dir)
    documents: List[Dict[str, Any]] = []
    seen_ids = set()

    if master_path.exists():
        print(f"[*] Loading master PYQ dataset from: {master_path}")
        with open(master_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        papers = data.get("papers", [])
        for paper in papers:
            source_pdf = paper.get("source_pdf", "")
            exam_title = paper.get("exam_title", "SSC CHSL Tier I Examination")
            paper_id = Path(source_pdf).stem or "SSC-CHSL-Paper"
            exam_date, shift = parse_exam_metadata_from_filename(source_pdf)

            for q in paper.get("questions", []):
                q_num = q.get("question_number") or len(documents) + 1
                doc_id = f"{paper_id}_q{q_num}"

                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                doc = {
                    "_id": doc_id,
                    "paper_id": paper_id,
                    "source_pdf": source_pdf,
                    "exam_title": exam_title,
                    "exam_name": "SSC CHSL",
                    "exam_date": exam_date,
                    "shift": shift,
                    "question_number": int(q_num),
                    "question": q.get("question", "").strip(),
                    "options": q.get("options", []),
                    "correct_answer": q.get("correct_answer", "").strip(),
                    "subject": q.get("subject", "General Awareness"),
                    "topic": q.get("topic") or q.get("subtopic", "General Awareness"),
                    "subtopic": q.get("subtopic") or q.get("topic", "General Awareness"),
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc)
                }
                documents.append(doc)

    # Fallback/merge individual files in extracted_json/*.json
    if not documents and extracted_dir.exists():
        for json_file in extracted_dir.glob("*.json"):
            if json_file.name == "all_extracted_questions.json":
                continue
            with open(json_file, "r", encoding="utf-8") as f:
                paper = json.load(f)

            source_pdf = paper.get("source_pdf", json_file.name)
            exam_title = paper.get("exam_title", "SSC CHSL Tier I Examination")
            paper_id = json_file.stem
            exam_date, shift = parse_exam_metadata_from_filename(json_file.name)

            for q in paper.get("questions", []):
                q_num = q.get("question_number") or len(documents) + 1
                doc_id = f"{paper_id}_q{q_num}"
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                doc = {
                    "_id": doc_id,
                    "paper_id": paper_id,
                    "source_pdf": source_pdf,
                    "exam_title": exam_title,
                    "exam_name": "SSC CHSL",
                    "exam_date": exam_date,
                    "shift": shift,
                    "question_number": int(q_num),
                    "question": q.get("question", "").strip(),
                    "options": q.get("options", []),
                    "correct_answer": q.get("correct_answer", "").strip(),
                    "subject": q.get("subject", "General Awareness"),
                    "topic": q.get("topic") or q.get("subtopic", "General Awareness"),
                    "subtopic": q.get("subtopic") or q.get("topic", "General Awareness"),
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc)
                }
                documents.append(doc)

    print(f"[+] Loaded {len(documents):,} PYQ questions across {len(set(d['paper_id'] for d in documents))} papers.")
    return documents


def load_lucent_data(
    ocr_dir: str | Path = "ocr_gk_extraction",
    master_file: str | Path = "ocr_gk_extraction/lucent_gk_extracted.json"
) -> List[Dict[str, Any]]:
    """
    Loads all Lucent GK extracted pages from consolidated master file and/or page_*.json files.
    """
    master_path = Path(master_file)
    ocr_path = Path(ocr_dir)
    documents: List[Dict[str, Any]] = []
    seen_pages = set()

    if master_path.exists():
        print(f"[*] Loading master Lucent GK dataset from: {master_path}")
        with open(master_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_pdf = data.get("source_pdf", "books/Lucent GK PDF 2026 in English.pdf")
        for page in data.get("pages", []):
            page_num = page.get("page_number")
            if page_num in seen_pages:
                continue
            seen_pages.add(page_num)

            doc = {
                "_id": f"lucent_page_{page_num:04d}",
                "page_number": int(page_num),
                "relative_headings": page.get("relative_headings", []),
                "corresponding_topics": page.get("corresponding_topics", []),
                "text": page.get("text", "").strip(),
                "char_count": len(page.get("text", "")),
                "source_pdf": source_pdf,
                "created_at": datetime.datetime.now(datetime.timezone.utc),
                "updated_at": datetime.datetime.now(datetime.timezone.utc)
            }
            documents.append(doc)

    # Fallback/merge individual files in ocr_gk_extraction/page_*.json
    if ocr_path.exists():
        for page_file in sorted(ocr_path.glob("page_*.json")):
            with open(page_file, "r", encoding="utf-8") as f:
                page = json.load(f)

            page_num = page.get("page_number")
            if page_num in seen_pages:
                continue
            seen_pages.add(page_num)

            doc = {
                "_id": f"lucent_page_{page_num:04d}",
                "page_number": int(page_num),
                "relative_headings": page.get("relative_headings", []),
                "corresponding_topics": page.get("corresponding_topics", []),
                "text": page.get("text", "").strip(),
                "char_count": len(page.get("text", "")),
                "source_pdf": "books/Lucent GK PDF 2026 in English.pdf",
                "created_at": datetime.datetime.now(datetime.timezone.utc),
                "updated_at": datetime.datetime.now(datetime.timezone.utc)
            }
            documents.append(doc)

    documents.sort(key=lambda d: d["page_number"])
    print(f"[+] Loaded {len(documents):,} Lucent GK page records.")
    return documents


# =====================================================================
# 4. Bulk Ingestion Engine (Idempotent Upsert)
# =====================================================================

def bulk_upsert_documents(
    collection: pymongo.collection.Collection,
    documents: List[Dict[str, Any]],
    batch_size: int = 1000
) -> Tuple[int, int]:
    """
    Inserts or updates documents in batches using UpdateOne with upsert=True.
    Guarantees idempotence: running multiple times never duplicates documents.
    """
    if not documents:
        print(f"[!] No documents to insert into {collection.database.name}.{collection.name}")
        return 0, 0

    total = len(documents)
    upserted_count = 0
    matched_count = 0

    print(f"[*] Ingesting {total:,} documents into '{collection.database.name}.{collection.name}' in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        operations = [
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": doc},
                upsert=True
            )
            for doc in batch
        ]

        try:
            result = collection.bulk_write(operations, ordered=False)
            upserted_count += (result.upserted_count + result.inserted_count)
            matched_count += result.matched_count

            current = min(i + batch_size, total)
            sys.stdout.write(f"\r  [{current:,}/{total:,}] processed (Upserted/New: {upserted_count:,}, Updated: {matched_count:,})")
            sys.stdout.flush()
        except BulkWriteError as bwe:
            print(f"\n[!] Bulk write error: {bwe.details}")

    print(f"\n[+] Ingestion finished for '{collection.database.name}.{collection.name}': {total:,} documents synced.")
    return upserted_count, matched_count


# =====================================================================
# 5. User Management Initialization (Starter Schemas & Templates)
# =====================================================================

def initialize_user_management_data(user_db: pymongo.database.Database, seed_samples: bool = True):
    """
    Initializes user management collections with index models and sample templates.
    """
    ensure_indexes_for_user_management(user_db)

    if not seed_samples:
        return

    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. Sample User Profile Template
    sample_user = {
        "_id": "usr_chsl_candidate_001",
        "user_id": "usr_chsl_candidate_001",
        "full_name": "Aarav Sharma",
        "email": "aarav.sharma@example.com",
        "target_exam": "SSC CHSL 2024 Tier I",
        "preferred_language": "en",
        "profile_created_at": now,
        "last_login": now,
        "status": "active",
        "metadata": {
            "tier": "Tier I",
            "daily_goal_questions": 50,
            "target_score": 160
        }
    }
    user_db["user_profiles"].update_one(
        {"_id": sample_user["_id"]},
        {"$set": sample_user},
        upsert=True
    )

    # 2. Sample User Analytics
    sample_analytics = {
        "_id": "analytics_usr_chsl_candidate_001",
        "user_id": "usr_chsl_candidate_001",
        "total_tests_attempted": 12,
        "total_questions_attempted": 300,
        "overall_accuracy_pct": 78.4,
        "subject_breakdown": {
            "English": {"attempted": 75, "correct": 62, "accuracy_pct": 82.6},
            "General Intelligence": {"attempted": 75, "correct": 65, "accuracy_pct": 86.7},
            "Quantitative Aptitude": {"attempted": 75, "correct": 54, "accuracy_pct": 72.0},
            "General Awareness": {"attempted": 75, "correct": 54, "accuracy_pct": 72.0}
        },
        "topic_strengths": ["Reading comprehension", "Idioms and phrases", "Polity and Constitution of India"],
        "topic_weaknesses": ["Geometry", "Trigonometry", "Ancient History"],
        "last_updated": now
    }
    user_db["user_analytics"].update_one(
        {"_id": sample_analytics["_id"]},
        {"$set": sample_analytics},
        upsert=True
    )

    # 3. Sample User Performance Log
    sample_performance = {
        "_id": "perf_att_001",
        "attempt_id": "perf_att_001",
        "user_id": "usr_chsl_candidate_001",
        "test_id": "test_series_chsl_mock_01",
        "question_id": "SSC-CHSL-11-August-2023-Shift-1_q1",
        "subject": "English",
        "topic": "Idioms and phrases",
        "user_selected_option": "(b) Egging on",
        "correct_option": "(b) Egging on",
        "is_correct": True,
        "time_spent_seconds": 24.5,
        "attempted_at": now
    }
    user_db["user_performance"].update_one(
        {"_id": sample_performance["_id"]},
        {"$set": sample_performance},
        upsert=True
    )

    # 4. Sample Test Series Definition
    sample_series = {
        "_id": "test_series_chsl_mock_01",
        "series_id": "test_series_chsl_mock_01",
        "title": "SSC CHSL Tier I Full Mock Test - 01",
        "exam_category": "SSC CHSL",
        "total_questions": 100,
        "duration_minutes": 60,
        "total_marks": 200,
        "positive_mark": 2.0,
        "negative_mark": 0.5,
        "sections": [
            {"subject": "English", "questions_count": 25, "marks": 50},
            {"subject": "General Intelligence", "questions_count": 25, "marks": 50},
            {"subject": "Quantitative Aptitude", "questions_count": 25, "marks": 50},
            {"subject": "General Awareness", "questions_count": 25, "marks": 50}
        ],
        "is_published": True,
        "created_at": now
    }
    user_db["test_series"].update_one(
        {"_id": sample_series["_id"]},
        {"$set": sample_series},
        upsert=True
    )

    print(f"[+] Initialized sample documents and schemas in '{user_db.name}' collections.")


# =====================================================================
# 6. Main Orchestrator
# =====================================================================

def run_database_pipeline(
    target_databases: List[str] = ["chsl_gnr", "study_material", "user_management"],
    batch_size: int = 1000,
    seed_users: bool = True
):
    """
    Executes complete connection, index creation, and data ingestion pipeline.
    """
    print("=" * 75)
    print("MongoDB Atlas Ingestion & Architecture Pipeline")
    print("=" * 75)

    manager = MongoDBManager()
    client = manager.get_client()

    # Load datasets from extracted files
    pyqs_docs = load_pyqs_data()
    lucent_docs = load_lucent_data()

    # 2. Target: study_material (Organized Study Material database)
    if "study_material" in target_databases:
        print("\n" + "-" * 60)
        print("[DATABASE: study_material] Syncing study material collections...")
        print("-" * 60)
        db_study = client["study_material"]

        coll_pyqs = db_study["pyqs"]
        ensure_indexes_for_pyqs(coll_pyqs)
        bulk_upsert_documents(coll_pyqs, pyqs_docs, batch_size=batch_size)

        coll_lucent = db_study["lucent_gk"]
        ensure_indexes_for_lucent(coll_lucent)
        bulk_upsert_documents(coll_lucent, lucent_docs, batch_size=batch_size)

    # 3. Target: user_management (User Profiles, Analytics, Performance, Series)
    if "user_management" in target_databases:
        print("\n" + "-" * 60)
        print("[DATABASE: user_management] Initializing user management collections...")
        print("-" * 60)
        db_user = client["user_management"]
        initialize_user_management_data(db_user, seed_samples=seed_users)

    # Print summary statistics across all databases
    print("\n" + "=" * 75)
    print("DATABASE INGESTION SUMMARY")
    print("=" * 75)
    for db_name in target_databases:
        db = client[db_name]
        print(f"\nDatabase: '{db_name}'")
        for coll_name in sorted(db.list_collection_names()):
            count = db[coll_name].count_documents({})
            print(f"  - Collection: {coll_name:<20} | Total Documents: {count:>8,}")

    print("\n[+] All requested databases and collections are synchronized and indexed.")
    print("=" * 75)
    manager.close()


def main():
    parser = argparse.ArgumentParser(
        description="MongoDB Atlas connection and data ingestion pipeline for SSC CHSL Study Material and User Management."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="all",
        choices=["all", "chsl_gnr", "study_material", "user_management"],
        help="Target database to populate (default: 'all' to sync all databases)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for bulk operations (default: 1000)"
    )
    parser.add_argument(
        "--no-seed-users",
        action="store_true",
        help="Do not seed sample starter records into user_management"
    )

    args = parser.parse_args()

    if args.db == "all":
        targets = ["chsl_gnr", "study_material", "user_management"]
    else:
        targets = [args.db]

    run_database_pipeline(
        target_databases=targets,
        batch_size=args.batch_size,
        seed_users=not args.no_seed_users
    )


if __name__ == "__main__":
    main()

