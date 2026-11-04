"""
PDF Question Extractor
======================
Extracts questions, multiple-choice options, and correct answers from SSC/competitive exam PDFs.
Strictly validates and formats output into JSON using Pydantic schemas.

Supports both:
1. Native high-accuracy layout & visual parser (PyMuPDF):
   - Accurately parses practice/mock papers with text answer keys: `Ans.(a/b/c/d)`
   - Accurately parses official TCS iON response sheets by detecting the embedded 16x16 green tick icons.
2. Optional Google GenAI (Gemini) structured output extraction via `google-genai` SDK.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import List, Optional

import pymupdf  # PyMuPDF
from pydantic import BaseModel, Field

# Optional Google GenAI SDK support
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    genai = None
    types = None
    HAS_GENAI = False


# ==========================================
# 1. Pydantic Output Schemas
# ==========================================

class Question(BaseModel):
    """Schema representing an individual multiple-choice question."""
    question_number: Optional[int] = Field(
        default=None,
        description="The sequence number of the question in the paper."
    )
    question: str = Field(
        ...,
        description="The question prompt or problem statement."
    )
    options: List[str] = Field(
        ...,
        description="List of answer options (e.g. Option 1 to 4 or (a) to (d))."
    )
    correct_answer: str = Field(
        ...,
        description="The correct answer text or option label."
    )
    subject: Optional[str] = Field(
        default=None,
        description="Broad subject category according to topic.txt."
    )
    subtopic: Optional[str] = Field(
        default=None,
        description="Specific subtopic according to topic.txt."
    )
    topic: Optional[str] = Field(
        default=None,
        description="Alias for subtopic according to topic.txt."
    )



class Quiz(BaseModel):
    """Schema representing an entire quiz or exam extracted from a PDF."""
    source_pdf: str = Field(
        ...,
        description="Name or file path of the source PDF document."
    )
    exam_title: Optional[str] = Field(
        default=None,
        description="Title or subject header of the examination."
    )
    total_questions: int = Field(
        default=0,
        description="Total number of valid questions extracted."
    )
    questions: List[Question] = Field(
        default_factory=list,
        description="List of all extracted multiple-choice questions."
    )


class QuizDataset(BaseModel):
    """Schema representing the collection of all extracted quizzes from a folder."""
    total_papers: int = Field(
        default=0,
        description="Total number of PDF question papers processed."
    )
    total_questions: int = Field(
        default=0,
        description="Total count of questions across all processed papers."
    )
    papers: List[Quiz] = Field(
        default_factory=list,
        description="List of quizzes extracted from each PDF."
    )


# ==========================================
# 2. Native PDF Parsers
# ==========================================

def _clean_text(text: str) -> str:
    """Normalize whitespace and strip unwanted non-breaking characters."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\r", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _parse_similar_paper_pdf(doc: pymupdf.Document, filename: str) -> Quiz:
    """
    Parses 'Similar Paper' style PDFs where answers are marked in text:
    e.g.
        Q1. What is the capital of India?
        (a) Mumbai
        (b) New Delhi
        (c) Kolkata
        (d) Chennai
        Ans.(b)
    """
    full_text = "\n".join([page.get_text() for page in doc])

    # Extract exam title from first page if present
    exam_title = None
    first_lines = [l.strip() for l in doc[0].get_text().splitlines() if l.strip()]
    for line in first_lines[:5]:
        if "SSC" in line or "Paper" in line or "Exam" in line:
            exam_title = line
            break

    # Find question blocks starting with Q<num>. or Q<num>:
    q_matches = list(re.finditer(r"(?:^|\n)\s*(Q\d+[\.:].*?)(?=(?:^|\n)\s*Q\d+[\.:]|\Z)", full_text, re.DOTALL))
    extracted_questions: List[Question] = []

    for match in q_matches:
        block = match.group(1).strip()

        # Extract question number
        num_m = re.match(r"^Q(\d+)[\.:]", block)
        q_num = int(num_m.group(1)) if num_m else len(extracted_questions) + 1

        # Look for Ans.(x) or Ans: (x) or Ans. x
        ans_m = re.search(r"\bAns[\.:]?\s*\(?([a-dA-D1-4])\)?", block, re.IGNORECASE)
        content_before_ans = block[:ans_m.start()] if ans_m else block
        ans_label = ans_m.group(1).lower() if ans_m else ""

        # Extract options: (a) text, (b) text, (c) text, (d) text
        # Use [^\S\r\n]* so newlines are not consumed by whitespace matching
        opts_iter = list(re.finditer(r"(?:^|\n)[^\S\r\n]*\(([a-dA-D1-4])\)[^\S\r\n]*(.*?)(?=(?:^|\n)[^\S\r\n]*\([a-dA-D1-4]\)|\Z)", content_before_ans, re.DOTALL))

        options_dict: dict[str, str] = {}
        first_opt_start = len(content_before_ans)

        if opts_iter:
            first_opt_start = opts_iter[0].start()
            for om in opts_iter:
                key = om.group(1).lower()
                val = _clean_text(om.group(2))
                options_dict[key] = val

        # Question prompt is everything before the first option
        q_prompt = content_before_ans[:first_opt_start].strip()
        q_prompt = re.sub(r"^Q\d+[\.:]\s*", "", q_prompt).strip()
        q_prompt = _clean_text(q_prompt)

        # Standardize options into 4 items (a, b, c, d)
        labels = ["a", "b", "c", "d"]
        options_list = [
            f"({lbl}) {options_dict.get(lbl, '')}".strip()
            for lbl in labels
        ]

        # Determine correct answer text
        correct_answer_text = ""
        if ans_label in options_dict and options_dict[ans_label]:
            correct_answer_text = f"({ans_label}) {options_dict[ans_label]}"
        elif ans_label:
            correct_answer_text = f"({ans_label})"

        extracted_questions.append(
            Question(
                question_number=q_num,
                question=q_prompt or f"Question {q_num}",
                options=options_list,
                correct_answer=correct_answer_text
            )
        )

    return Quiz(
        source_pdf=filename,
        exam_title=exam_title or Path(filename).stem,
        total_questions=len(extracted_questions),
        questions=extracted_questions
    )


def _parse_tcs_response_sheet_pdf(doc: pymupdf.Document, filename: str) -> Quiz:
    """
    Parses official TCS iON / SSC Response Sheets where:
    - Questions start with 'Q.<num>'
    - Options follow 'Ans\\n1. ...\\n2. ...\\n3. ...\\n4. ...'
    - Correct answer is marked by an embedded 16x16 green tick icon at the option's y-coordinate.
    """
    extracted_questions: List[Question] = []
    exam_title = None

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # Read exam header from the first page
        if page_idx == 0 and not exam_title:
            for blk in page.get_text("blocks"):
                if "Combined Higher Secondary" in blk[4] or "Tier" in blk[4]:
                    exam_title = _clean_text(blk[4].splitlines()[0])
                    break

        # Detect green checkmark bounding boxes (16x16 icons)
        ticks = [
            item["bbox"]
            for item in page.get_image_info()
            if item.get("width") == 16 and item.get("height") == 16
        ]
        ticks.sort(key=lambda b: b[1])

        # Get text blocks ordered vertically
        blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))

        # Find question boundaries (Q.<num>)
        q_indices = []
        for idx, b in enumerate(blocks):
            m = re.search(r"\bQ\.(\d+)", b[4])
            if m:
                q_indices.append((idx, int(m.group(1)), b))

        for i, (idx, q_num, b) in enumerate(q_indices):
            next_idx = q_indices[i + 1][0] if i + 1 < len(q_indices) else len(blocks)
            q_blocks = blocks[idx:next_idx]

            # Find ticks lying within the vertical boundary of this question
            y_min = q_blocks[0][1] - 5
            y_max = q_blocks[-1][3] + 5
            q_ticks = [t for t in ticks if y_min <= t[1] <= y_max]

            q_text_lines: List[str] = []
            ans_lines: List[str] = []
            is_ans = False

            for blk in q_blocks:
                text = blk[4]
                # Skip header/footer metadata
                if any(hdr in text for hdr in ["Combined Higher Secondary", "Exam Date", "Exam Time", "Section :"]):
                    continue

                for line in text.split("\n"):
                    l_str = line.strip()
                    if not l_str:
                        continue
                    if l_str == "Ans" or l_str.startswith("Ans\n") or l_str.startswith("Ans ") or l_str.startswith("Ans\xa0"):
                        is_ans = True
                        rem = re.sub(r"^Ans[\s\xa0]*", "", l_str)
                        if rem:
                            ans_lines.append(rem)
                    elif is_ans:
                        ans_lines.append(l_str)
                    else:
                        q_text_lines.append(l_str)

            # Clean question text
            full_q = _clean_text(" ".join(q_text_lines))
            full_q = re.sub(r"^\s*Q\.\d+\s*", "", full_q).strip()

            # Parse numbered options 1. 2. 3. 4.
            joined_ans = "\n".join(ans_lines)
            opt_matches = list(re.finditer(r"(?:^|\n)\s*([1-4])\.\s*(.*?)(?=(?:^|\n)\s*[1-4]\.|\Z)", joined_ans, re.DOTALL))
            options_dict: dict[int, str] = {}
            for om in opt_matches:
                opt_idx = int(om.group(1))
                val = _clean_text(om.group(2))
                options_dict[opt_idx] = val if val else f"Option {opt_idx}"

            # Match tick image y-coordinate to the closest option block
            correct_opt = None
            if q_ticks:
                tick_y = q_ticks[0][1]
                best_diff = 9999.0
                best_num = None
                for blk in q_blocks:
                    m_opt = re.match(r"^\s*(?:Ans[\s\xa0]+)?([1-4])\.", blk[4])
                    if m_opt:
                        opt_n = int(m_opt.group(1))
                        diff = abs(blk[1] - tick_y)
                        if diff < best_diff:
                            best_diff = diff
                            best_num = opt_n
                correct_opt = best_num

            # Format 4 options
            opts_list = [
                f"{k}. {options_dict.get(k, f'Option {k}')}"
                for k in [1, 2, 3, 4]
            ]

            corr_ans_text = ""
            if correct_opt:
                val = options_dict.get(correct_opt, f"Option {correct_opt}")
                corr_ans_text = f"{correct_opt}. {val}"

            extracted_questions.append(
                Question(
                    question_number=len(extracted_questions) + 1,
                    question=full_q or f"Question {len(extracted_questions) + 1}",
                    options=opts_list,
                    correct_answer=corr_ans_text
                )
            )

    return Quiz(
        source_pdf=filename,
        exam_title=exam_title or Path(filename).stem,
        total_questions=len(extracted_questions),
        questions=extracted_questions
    )


def extract_questions_from_pdf(pdf_path: str | Path) -> Quiz:
    """
    Main extraction function for a single PDF.
    Auto-detects paper format and extracts questions, options, and correct answers.
    Returns a validated Pydantic Quiz model.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    filename = pdf_path.name

    # Determine paper format from first 2 pages
    first_pages_text = "\n".join([doc[p].get_text() for p in range(min(2, len(doc)))])

    if "Ans.(" in first_pages_text or "Ans. (" in first_pages_text:
        # Similar Paper format
        return _parse_similar_paper_pdf(doc, filename)
    else:
        # TCS iON Official Response Sheet format
        return _parse_tcs_response_sheet_pdf(doc, filename)


# ==========================================
# 3. Optional Google GenAI LLM Integration
# ==========================================

def extract_questions_with_gemini(
    pdf_path: str | Path,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash"
) -> Quiz:
    """
    Extracts questions and answers using Google GenAI SDK (Gemini) with Pydantic structured output.
    Requires: `pip install google-genai` and `GEMINI_API_KEY`.
    """
    if not HAS_GENAI:
        raise ImportError(
            "google-genai is not installed. Run `pip install google-genai` "
            "or use the native `extract_questions_from_pdf` parser."
        )

    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    pdf_path = Path(pdf_path)

    # Extract text from all pages
    doc = pymupdf.open(str(pdf_path))
    pdf_text = "\n".join([f"--- Page {i+1} ---\n" + page.get_text() for i, page in enumerate(doc)])

    prompt = (
        "You are an expert exam parser. Extract all multiple-choice questions from the following exam text. "
        "For each question, extract: question prompt, all 4 options, and the correct answer. "
        f"The exam source file is '{pdf_path.name}'.\n\n"
        f"EXAM TEXT:\n{pdf_text}"
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Quiz,
        ),
    )

    # Validate strictly into Pydantic Quiz schema
    return Quiz.model_validate_json(response.text)


# ==========================================
# 4. Folder Processing & Batch JSON Export
# ==========================================

def extract_from_folder(
    folder_path: str | Path,
    output_dir: Optional[str | Path] = "extracted_json",
    save_individual: bool = True,
    save_combined: bool = True
) -> QuizDataset:
    """
    Extracts questions from all PDF files in `folder_path`, strictly validates
    them into Pydantic models, and writes formatted JSON output files.

    Args:
        folder_path: Path to the directory containing PDFs.
        output_dir: Path to directory where JSON files will be written.
        save_individual: If True, writes a separate JSON file for each PDF.
        save_combined: If True, writes a single consolidated JSON file for all PDFs.

    Returns:
        QuizDataset: Pydantic model containing all extracted quizzes.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Directory not found: {folder}")

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder.resolve()}")
        return QuizDataset()

    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(pdf_files)} PDF(s) in: {folder.resolve()}")
    print("-" * 65)

    all_quizzes: List[Quiz] = []
    total_qs_extracted = 0

    for idx, pdf in enumerate(pdf_files, 1):
        try:
            quiz = extract_questions_from_pdf(pdf)
            all_quizzes.append(quiz)
            total_qs_extracted += quiz.total_questions

            print(f"[{idx:02d}/{len(pdf_files):02d}] {pdf.name[:45]:<45} | Extracted: {quiz.total_questions:3d} Qs")

            # Save individual JSON per PDF
            if save_individual and out_path:
                json_filename = out_path / f"{pdf.stem}.json"
                with open(json_filename, "w", encoding="utf-8") as f:
                    # Strict formatting with Pydantic model_dump_json
                    f.write(quiz.model_dump_json(indent=2))

        except Exception as e:
            print(f"[{idx:02d}/{len(pdf_files):02d}] Error processing {pdf.name}: {e}")

    # Build root collection model
    dataset = QuizDataset(
        total_papers=len(all_quizzes),
        total_questions=total_qs_extracted,
        papers=all_quizzes
    )

    # Save combined dataset JSON
    if save_combined and out_path:
        combined_file = out_path / "all_extracted_questions.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            f.write(dataset.model_dump_json(indent=2))
        print(f"\n[+] Combined JSON saved to: {combined_file.resolve()}")

    print("=" * 65)
    print(f"Extraction Completed!")
    print(f"Total Papers Processed: {dataset.total_papers}")
    print(f"Total Questions Formatted: {dataset.total_questions}")
    if out_path:
        print(f"JSON Output Directory: {out_path.resolve()}")
    print("=" * 65)

    return dataset


# ==========================================
# 5. CLI Execution
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract questions, options, and answers from PDFs into strict Pydantic JSON."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="chsl_pyqs",
        help="Path to folder containing PDF files (default: 'chsl_pyqs')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="extracted_json",
        help="Output directory to save formatted JSON files (default: 'extracted_json')"
    )

    args = parser.parse_args()
    extract_from_folder(args.folder, args.output)
