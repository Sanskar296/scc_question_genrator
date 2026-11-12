"""
ChromaDB Vector Embeddings & Semantic Retrieval Pipeline
========================================================
Initializes a local persistent ChromaDB client for vector search and indexing.
Loads structured datasets:
  1. SSC CHSL Previous Year Questions (PYQs): 'extracted_json/all_extracted_questions.json'
  2. Lucent General Knowledge (GK) Book Pages: 'ocr_gk_extraction/lucent_gk_extracted.json'

Generates 384-dimensional dense vector embeddings using all-MiniLM-L6-v2 ONNX.
Indexes text along with essential metadata:
  - Page numbers, headings, topics, subtopics, exam dates, shifts, and paper IDs.
Enables fast semantic retrieval across individual and unified study material collections.

Author: Antigravity (Expert Vector Search System)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# UTF-8 encoding safeguard for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# =====================================================================
# 1. ChromaDB Cloud Credentials & Thin-Client Bootstrap Safeguard
# =====================================================================

DEFAULT_CHROMA_API_KEY = os.getenv("CHROMADB_API_KEY", "ck-8AARJhYU7tP9H7s4Q41UfCHebnR8td6NkapQMgcuCj6Y")
DEFAULT_CHROMA_TENANT = os.getenv("CHROMADB_TENANT", "2d8eb3d4-29d6-4479-8c3b-3efc4ccd659c")
DEFAULT_CHROMA_DATABASE = os.getenv("CHROMADB_DATABASE", "chsl_embeddings")

# In environments where chromadb-client was co-installed with chromadb,
# patch is_thin_client to False in sys.modules before importing chromadb.
if "chromadb.is_thin_client" not in sys.modules:
    _m = types.ModuleType("chromadb.is_thin_client")
    _m.is_thin_client = False
    sys.modules["chromadb.is_thin_client"] = _m

import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as ef


# =====================================================================
# 2. Metadata Parsing Utilities
# =====================================================================

def parse_exam_metadata_from_filename(filename: str) -> Tuple[str, str]:
    """Extracts date and shift from exam PDF/JSON filename."""
    date_match = re.search(r"(\d{1,2})[- ]([A-Za-z]+)[- ](\d{4})", filename)
    date_str = f"{date_match.group(1)} {date_match.group(2).capitalize()} {date_match.group(3)}" if date_match else "Unknown Date"

    shift_match = re.search(r"shift[- ]?(\d+)", filename, re.IGNORECASE)
    shift_str = f"Shift {shift_match.group(1)}" if shift_match else "Unknown Shift"

    return date_str, shift_str


# =====================================================================
# 3. Dataset Loaders for Vector Embedding
# =====================================================================

def _resolve_data_path(primary: str | Path, alt_dir: str = "client/dist/assets") -> Path:
    p = Path(primary)
    if p.exists():
        return p
    alt = Path(alt_dir) / primary
    if alt.exists():
        return alt
    return p

def load_pyqs_for_embedding(
    master_file: str | Path = "client/dist/assets/extracted_json/all_extracted_questions.json",
    extracted_dir: str | Path = "client/dist/assets/extracted_json",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Loads PYQ questions from JSON and formats them for vector embedding.
    """
    master_path = _resolve_data_path(master_file)
    extracted_path = _resolve_data_path(extracted_dir)
    items: List[Dict[str, Any]] = []
    seen_ids = set()

    if master_path.exists():
        print(f"[*] Loading PYQs from master file: {master_path}")
        with open(master_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for paper in data.get("papers", []):
            source_pdf = paper.get("source_pdf", "SSC-CHSL-Paper.pdf")
            paper_id = Path(source_pdf).stem
            exam_title = paper.get("exam_title", "Combined Higher Secondary Level Examination")
            exam_date, shift = parse_exam_metadata_from_filename(source_pdf)

            for q in paper.get("questions", []):
                q_num = q.get("question_number") or (len(items) + 1)
                doc_id = f"pyq_{paper_id}_q{q_num}"

                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                question_text = q.get("question", "").strip()
                options = q.get("options", [])
                options_str = "\n".join(options) if options else "N/A"
                subject = q.get("subject", "General Awareness")
                topic = q.get("topic") or q.get("subtopic", "General Awareness")
                subtopic = q.get("subtopic") or q.get("topic", "General Awareness")
                correct_answer = q.get("correct_answer", "").strip()

                # Semantic text representation for dense retrieval
                embed_text = (
                    f"Subject: {subject}\n"
                    f"Topic: {topic}\n"
                    f"Exam: {exam_title} | {shift} | Date: {exam_date}\n"
                    f"Question: {question_text}\n"
                    f"Options:\n{options_str}"
                ).strip()

                # Essential metadata (strict primitive types for ChromaDB)
                metadata = {
                    "dataset": "chsl_pyqs",
                    "doc_type": "question",
                    "question_id": doc_id,
                    "question_number": int(q_num),
                    "subject": str(subject),
                    "topic": str(topic),
                    "subtopic": str(subtopic),
                    "shift": str(shift),
                    "exam_date": str(exam_date),
                    "exam_title": str(exam_title),
                    "paper_id": str(paper_id),
                    "source_pdf": str(source_pdf),
                    "correct_answer": str(correct_answer)[:200]
                }

                items.append({
                    "id": doc_id,
                    "document": embed_text,
                    "metadata": metadata
                })

                if limit and len(items) >= limit:
                    break
            if limit and len(items) >= limit:
                break

    # Fallback to individual files if master was empty
    if not items and extracted_path.exists():
        for json_file in extracted_path.glob("*.json"):
            if json_file.name == "all_extracted_questions.json":
                continue
            with open(json_file, "r", encoding="utf-8") as f:
                paper = json.load(f)

            source_pdf = paper.get("source_pdf", json_file.name)
            paper_id = json_file.stem
            exam_title = paper.get("exam_title", "Combined Higher Secondary Level Examination")
            exam_date, shift = parse_exam_metadata_from_filename(json_file.name)

            for q in paper.get("questions", []):
                q_num = q.get("question_number") or (len(items) + 1)
                doc_id = f"pyq_{paper_id}_q{q_num}"
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                question_text = q.get("question", "").strip()
                options = q.get("options", [])
                options_str = "\n".join(options) if options else "N/A"
                subject = q.get("subject", "General Awareness")
                topic = q.get("topic") or q.get("subtopic", "General Awareness")
                subtopic = q.get("subtopic") or q.get("topic", "General Awareness")

                embed_text = (
                    f"Subject: {subject}\n"
                    f"Topic: {topic}\n"
                    f"Exam: {exam_title} | {shift} | Date: {exam_date}\n"
                    f"Question: {question_text}\n"
                    f"Options:\n{options_str}"
                ).strip()

                metadata = {
                    "dataset": "chsl_pyqs",
                    "doc_type": "question",
                    "question_id": doc_id,
                    "question_number": int(q_num),
                    "subject": str(subject),
                    "topic": str(topic),
                    "subtopic": str(subtopic),
                    "shift": str(shift),
                    "exam_date": str(exam_date),
                    "exam_title": str(exam_title),
                    "paper_id": str(paper_id),
                    "source_pdf": str(source_pdf),
                    "correct_answer": str(q.get("correct_answer", ""))[:200]
                }

                items.append({
                    "id": doc_id,
                    "document": embed_text,
                    "metadata": metadata
                })

                if limit and len(items) >= limit:
                    break
            if limit and len(items) >= limit:
                break

    print(f"[+] Prepared {len(items):,} PYQ questions for vector embedding.")
    return items


def load_lucent_for_embedding(
    master_file: str | Path = "client/dist/assets/ocr_gk_extraction/lucent_gk_extracted.json",
    ocr_dir: str | Path = "client/dist/assets/ocr_gk_extraction",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Loads Lucent GK page records and formats them for vector embedding.
    """
    master_path = _resolve_data_path(master_file)
    ocr_path = _resolve_data_path(ocr_dir)
    items: List[Dict[str, Any]] = []
    seen_pages = set()

    if master_path.exists():
        print(f"[*] Loading Lucent GK pages from master file: {master_path}")
        with open(master_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_pdf = data.get("source_pdf", "books/Lucent GK PDF 2026 in English.pdf")
        for page in data.get("pages", []):
            page_num = page.get("page_number")
            if page_num in seen_pages:
                continue
            seen_pages.add(page_num)

            doc_id = f"lucent_page_{page_num:04d}"
            headings = page.get("relative_headings", [])
            topics = page.get("corresponding_topics", [])
            text = page.get("text", "").strip()

            headings_str = ", ".join(headings) if headings else "General Knowledge"
            topics_str = ", ".join(topics) if topics else "General Awareness"
            primary_topic = topics[0] if topics else "General Awareness"

            embed_text = (
                f"Source: Lucent General Knowledge | Page: {page_num}\n"
                f"Headings: {headings_str}\n"
                f"Topics: {topics_str}\n\n"
                f"{text}"
            ).strip()

            metadata = {
                "dataset": "lucent_gk",
                "doc_type": "book_page",
                "page_number": int(page_num),
                "relative_headings": str(headings_str)[:500],
                "topics": str(topics_str)[:500],
                "topic": str(primary_topic),
                "primary_topic": str(primary_topic),
                "source_pdf": str(source_pdf),
                "char_count": len(text),
                "shift": "N/A",
                "exam_date": "N/A"
            }

            items.append({
                "id": doc_id,
                "document": embed_text,
                "metadata": metadata
            })

            if limit and len(items) >= limit:
                break

    # Fallback to individual page_*.json files
    if not items and ocr_path.exists():
        for page_file in sorted(ocr_path.glob("page_*.json")):
            with open(page_file, "r", encoding="utf-8") as f:
                page = json.load(f)

            page_num = page.get("page_number")
            if page_num in seen_pages:
                continue
            seen_pages.add(page_num)

            doc_id = f"lucent_page_{page_num:04d}"
            headings = page.get("relative_headings", [])
            topics = page.get("corresponding_topics", [])
            text = page.get("text", "").strip()

            headings_str = ", ".join(headings) if headings else "General Knowledge"
            topics_str = ", ".join(topics) if topics else "General Awareness"
            primary_topic = topics[0] if topics else "General Awareness"

            embed_text = (
                f"Source: Lucent General Knowledge | Page: {page_num}\n"
                f"Headings: {headings_str}\n"
                f"Topics: {topics_str}\n\n"
                f"{text}"
            ).strip()

            metadata = {
                "dataset": "lucent_gk",
                "doc_type": "book_page",
                "page_number": int(page_num),
                "relative_headings": str(headings_str)[:500],
                "topics": str(topics_str)[:500],
                "topic": str(primary_topic),
                "primary_topic": str(primary_topic),
                "source_pdf": "books/Lucent GK PDF 2026 in English.pdf",
                "char_count": len(text),
                "shift": "N/A",
                "exam_date": "N/A"
            }

            items.append({
                "id": doc_id,
                "document": embed_text,
                "metadata": metadata
            })

            if limit and len(items) >= limit:
                break

    items.sort(key=lambda it: it["metadata"]["page_number"])
    print(f"[+] Prepared {len(items):,} Lucent GK pages for vector embedding.")
    return items


# =====================================================================
# 4. ChromaDB Client & Vector Indexer (Cloud & Local Support)
# =====================================================================

class VectorDBManager:
    """
    Manages ChromaDB collections, dense vector embeddings, and semantic queries.
    Supports both ChromaDB CloudClient and local PersistentClient with auto-fallback.
    """

    def __init__(
        self,
        db_path: str | Path = "./chroma_db",
        use_cloud: bool = True,
        api_key: Optional[str] = None,
        tenant: Optional[str] = None,
        database: Optional[str] = None
    ):
        self.db_path = Path(db_path).resolve()
        self.use_cloud = use_cloud
        self.api_key = api_key or os.getenv("CHROMADB_API_KEY", DEFAULT_CHROMA_API_KEY)
        self.tenant = tenant or os.getenv("CHROMADB_TENANT", DEFAULT_CHROMA_TENANT)
        self.database = database or os.getenv("CHROMADB_DATABASE", DEFAULT_CHROMA_DATABASE)

        # Local all-MiniLM-L6-v2 ONNX embedding function (384-dim, dense representation)
        self.embedding_function = ef.DefaultEmbeddingFunction()

        # 1. Connect to ChromaDB CloudClient if requested
        if self.use_cloud and self.api_key and self.tenant and self.database:
            masked_key = f"{self.api_key[:6]}...{self.api_key[-4:]}" if len(self.api_key) > 10 else "***"
            print(f"[*] Connecting to ChromaDB CloudClient:")
            print(f"    Tenant:   {self.tenant}")
            print(f"    Database: {self.database}")
            print(f"    API Key:  {masked_key}")
            try:
                self.client = chromadb.CloudClient(
                    api_key=self.api_key,
                    tenant=self.tenant,
                    database=self.database
                )
                print("[+] Successfully authenticated & connected to ChromaDB Cloud!")
            except Exception as e:
                print(f"[!] Warning: ChromaDB Cloud connection failed ({e}). Falling back to local PersistentClient.")
                self.use_cloud = False

        # 2. Local fallback or explicit local client
        if not self.use_cloud:
            self.db_path.mkdir(parents=True, exist_ok=True)
            print(f"[*] Initializing Local Persistent ChromaDB Client at: {self.db_path}")
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

        # Initialize collections with cosine distance metric for normalized ranking
        self.coll_pyqs = self.client.get_or_create_collection(
            name="chsl_pyqs",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine", "description": "SSC CHSL PYQs with shifts and dates"}
        )

        self.coll_lucent = self.client.get_or_create_collection(
            name="lucent_gk",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine", "description": "Lucent GK pages with headings and topics"}
        )

        self.coll_unified = self.client.get_or_create_collection(
            name="study_material",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine", "description": "Unified CHSL study material collection"}
        )

        storage_target = f"Chroma Cloud ({self.database})" if self.use_cloud else f"Local ({self.db_path})"
        print(f"[+] Active Vector Storage: {storage_target}")
        print(f"[+] Collections initialized: 'chsl_pyqs', 'lucent_gk', 'study_material'")

    def index_items_in_batches(
        self,
        collection: Any,
        items: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Generates vector embeddings and upserts documents in batches into ChromaDB Cloud or Local storage.
        Idempotent: updates existing vectors in-place if re-run.
        Includes automatic retry for resilient cloud network uploads.
        """
        if not items:
            return 0

        total = len(items)
        dest = f"Chroma Cloud ({self.database})" if self.use_cloud else "Local ChromaDB"
        print(f"\n[*] Indexing {total:,} documents into '{collection.name}' on {dest} in batches of {batch_size}...")

        start_time = time.time()
        for i in range(0, total, batch_size):
            batch = items[i:i + batch_size]
            b_ids = [item["id"] for item in batch]
            b_docs = [item["document"] for item in batch]
            b_metas = [item["metadata"] for item in batch]

            t_batch_start = time.time()
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    collection.upsert(
                        ids=b_ids,
                        documents=b_docs,
                        metadatas=b_metas
                    )
                    break
                except Exception as err:
                    if attempt < max_retries - 1:
                        backoff = 2 * (attempt + 1)
                        print(f"\n[!] Network glitch on batch ({i}-{i+len(batch)}), retrying in {backoff}s: {err}")
                        time.sleep(backoff)
                    else:
                        print(f"\n[X] Failed batch after {max_retries} attempts: {err}")
                        raise err

            elapsed_batch = time.time() - t_batch_start
            current = min(i + batch_size, total)
            rate = len(batch) / elapsed_batch if elapsed_batch > 0 else 0
            sys.stdout.write(f"\r  [{current:,}/{total:,}] embedded & stored (Batch: {elapsed_batch:4.2f}s | ~{rate:4.1f} docs/s)")
            sys.stdout.flush()

        total_elapsed = time.time() - start_time
        print(f"\n[+] Successfully indexed {total:,} items into '{collection.name}' on {dest} in {total_elapsed:.1f}s.")
        return total

    def query(
        self,
        query_text: str,
        collection_name: str = "chsl_pyqs",
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes semantic nearest-neighbor search with optional metadata filtering.
        """
        colls = {
            "chsl_pyqs": self.coll_pyqs,
            "lucent_gk": self.coll_lucent,
            "study_material": self.coll_unified
        }
        if collection_name not in colls:
            raise ValueError(f"Unknown collection '{collection_name}'. Choose from: {list(colls.keys())}")

        coll = colls[collection_name]
        results = coll.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        return results


# =====================================================================
# 5. Full Indexing & Search CLI Orchestrator
# =====================================================================

def run_indexing(
    db_path: str = "./chroma_db",
    use_cloud: bool = True,
    api_key: Optional[str] = None,
    tenant: Optional[str] = None,
    database: Optional[str] = None,
    batch_size: int = 100,
    limit_pyqs: Optional[int] = None,
    limit_lucent: Optional[int] = None,
    sync_unified: bool = True
):
    """
    Loads both datasets and executes complete Chroma vector indexing pipeline.
    """
    print("=" * 75)
    print("ChromaDB Vector Embeddings & Ingestion Engine (Cloud & Local)")
    print("=" * 75)

    vdb = VectorDBManager(
        db_path=db_path,
        use_cloud=use_cloud,
        api_key=api_key,
        tenant=tenant,
        database=database
    )

    # 1. Load data
    pyq_items = load_pyqs_for_embedding(limit=limit_pyqs)
    lucent_items = load_lucent_for_embedding(limit=limit_lucent)

    # 2. Index into specialized collections
    print("\n" + "-" * 60)
    print("[1/2] Indexing CHSL PYQs Dataset")
    print("-" * 60)
    vdb.index_items_in_batches(vdb.coll_pyqs, pyq_items, batch_size=batch_size)

    print("\n" + "-" * 60)
    print("[2/2] Indexing Lucent GK Dataset")
    print("-" * 60)
    vdb.index_items_in_batches(vdb.coll_lucent, lucent_items, batch_size=batch_size)

    # 3. Index into unified study material collection
    if sync_unified:
        print("\n" + "-" * 60)
        print("[3/3] Syncing Unified Study Material Collection")
        print("-" * 60)
        combined_items = pyq_items + lucent_items
        vdb.index_items_in_batches(vdb.coll_unified, combined_items, batch_size=batch_size)

    # Print summary statistics
    target_info = f"ChromaDB Cloud (Tenant: {vdb.tenant} | Database: {vdb.database})" if vdb.use_cloud else f"Local Directory: {vdb.db_path.resolve()}"
    print("\n" + "=" * 75)
    print("CHROMADB VECTOR INDEXING SUMMARY")
    print("=" * 75)
    print(f"Target Storage: {target_info}")
    print(f"  - Collection 'chsl_pyqs'      : {vdb.coll_pyqs.count():>7,} vectors")
    print(f"  - Collection 'lucent_gk'      : {vdb.coll_lucent.count():>7,} vectors")
    print(f"  - Collection 'study_material' : {vdb.coll_unified.count():>7,} vectors")
    print("=" * 75)


def run_query(
    query_text: str,
    collection_name: str = "chsl_pyqs",
    n_results: int = 5,
    db_path: str = "./chroma_db",
    use_cloud: bool = True,
    api_key: Optional[str] = None,
    tenant: Optional[str] = None,
    database: Optional[str] = None,
    filter_topic: Optional[str] = None,
    filter_shift: Optional[str] = None
):
    """
    Executes a vector search query against Cloud or Local ChromaDB and prints formatted results.
    """
    vdb = VectorDBManager(
        db_path=db_path,
        use_cloud=use_cloud,
        api_key=api_key,
        tenant=tenant,
        database=database
    )

    where_filter = {}
    if filter_topic:
        where_filter["topic"] = filter_topic
    if filter_shift:
        where_filter["shift"] = filter_shift
    if not where_filter:
        where_filter = None

    target = "ChromaDB Cloud" if vdb.use_cloud else "Local ChromaDB"
    print(f"\n[*] Searching '{collection_name}' on {target} for: \"{query_text}\" (n={n_results})")
    if where_filter:
        print(f"[*] Applied Metadata Filter: {where_filter}")

    results = vdb.query(
        query_text=query_text,
        collection_name=collection_name,
        n_results=n_results,
        where_filter=where_filter
    )

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    print("\n" + "=" * 75)
    print(f"TOP {len(ids)} SEMANTIC SEARCH RESULTS ({target})")
    print("=" * 75)

    for idx, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas), 1):
        dist = distances[idx - 1] if idx - 1 < len(distances) else 0.0
        similarity = 1.0 - dist if dist <= 1.0 else 0.0

        print(f"\n[Result #{idx}] ID: {doc_id} | Similarity: {similarity:.4f} (Distance: {dist:.4f})")
        if meta.get("dataset") == "chsl_pyqs":
            print(f"  Subject: {meta.get('subject')} | Topic: {meta.get('topic')}")
            print(f"  Exam: {meta.get('exam_title')} | Shift: {meta.get('shift')} | Date: {meta.get('exam_date')}")
            print(f"  Correct Answer: {meta.get('correct_answer')}")
        else:
            print(f"  Page: {meta.get('page_number')} | Topic: {meta.get('topic')}")
            print(f"  Headings: {meta.get('relative_headings')}")

        snippet = doc[:250].replace("\n", " ")
        print(f"  Snippet: {snippet}...")

    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="ChromaDB vector embedding generator and semantic search indexer for SSC CHSL study materials (Cloud & Local)."
    )
    parser.add_argument(
        "--cloud",
        dest="use_cloud",
        action="store_true",
        default=True,
        help="Store vectors in ChromaDB Cloud (default: True)"
    )
    parser.add_argument(
        "--local",
        dest="use_cloud",
        action="store_false",
        help="Store vectors in local ChromaDB PersistentClient"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="ChromaDB Cloud API key (default: from .env or DEFAULT_CHROMA_API_KEY)"
    )
    parser.add_argument(
        "--tenant",
        type=str,
        default=None,
        help="ChromaDB Cloud tenant ID (default: from .env or DEFAULT_CHROMA_TENANT)"
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="ChromaDB Cloud database name (default: from .env or DEFAULT_CHROMA_DATABASE)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./chroma_db",
        help="Local directory to store persistent ChromaDB data when in local mode (default: './chroma_db')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for generating embeddings & uploads (default: 100)"
    )
    parser.add_argument(
        "--limit-pyqs",
        type=int,
        default=None,
        help="Limit number of PYQ questions to index (default: None for all)"
    )
    parser.add_argument(
        "--limit-lucent",
        type=int,
        default=None,
        help="Limit number of Lucent GK pages to index (default: None for all)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query text to search across the vector index"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="study_material",
        choices=["chsl_pyqs", "lucent_gk", "study_material"],
        help="Collection to search (default: 'study_material')"
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of query results to return (default: 5)"
    )
    parser.add_argument(
        "--filter-topic",
        type=str,
        default=None,
        help="Filter search results by topic metadata"
    )
    parser.add_argument(
        "--filter-shift",
        type=str,
        default=None,
        help="Filter search results by shift metadata"
    )

    args = parser.parse_args()

    if args.query:
        run_query(
            query_text=args.query,
            collection_name=args.collection,
            n_results=args.n_results,
            db_path=args.db_path,
            use_cloud=args.use_cloud,
            api_key=args.api_key,
            tenant=args.tenant,
            database=args.database,
            filter_topic=args.filter_topic,
            filter_shift=args.filter_shift
        )
    else:
        run_indexing(
            db_path=args.db_path,
            use_cloud=args.use_cloud,
            api_key=args.api_key,
            tenant=args.tenant,
            database=args.database,
            batch_size=args.batch_size,
            limit_pyqs=args.limit_pyqs,
            limit_lucent=args.limit_lucent
        )


if __name__ == "__main__":
    main()


