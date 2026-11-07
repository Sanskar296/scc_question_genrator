"""
PDF Question & Book Content Extractor
======================================
1. Lucent GK Book Parser (OCR & Topic Matcher):
   - Scans Lucent GK PDF page-by-page from the 'books/' folder.
   - Utilizes EasyOCR with layout line-clustering and heading detection.
   - Matches page content and headings against the 14 General Awareness topics in topic.txt.
   - Saves strictly validated JSON entries into 'ocr_gk_extraction/'.

2. Legacy Question Paper Parser (PyMuPDF):
   - Extracts MCQs, options, and answer keys from competitive exam PDFs.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import pymupdf  # PyMuPDF
from pydantic import BaseModel, Field

# UTF-8 encoding safeguard for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Optional EasyOCR & Torch setup - lazily loaded to prevent initialization overhead
torch = None
easyocr = None
HAS_EASYOCR = False

def _ensure_easyocr():
    global torch, easyocr, HAS_EASYOCR
    if torch is None or easyocr is None:
        try:
            import torch as _torch
            import easyocr as _easyocr
            torch = _torch
            easyocr = _easyocr
            HAS_EASYOCR = True
        except ImportError:
            HAS_EASYOCR = False
    return HAS_EASYOCR


# ==========================================
# 1. Pydantic Schemas for Lucent GK Extraction
# ==========================================

class LucentGKPage(BaseModel):
    """Schema representing an individual extracted page from Lucent GK."""
    page_number: int = Field(
        ...,
        description="1-indexed physical page number of the book."
    )
    relative_headings: List[str] = Field(
        default_factory=list,
        description="Section, chapter, or sub-headings found on or relevant to this page."
    )
    corresponding_topics: List[str] = Field(
        default_factory=list,
        description="Matched General Awareness topics from topic.txt."
    )
    text: str = Field(
        ...,
        description="Full extracted text of the page."
    )


class LucentGKBookExtraction(BaseModel):
    """Schema representing the overall book extraction dataset."""
    source_pdf: str = Field(
        ...,
        description="Name or file path of the source Lucent GK PDF."
    )
    total_pages: int = Field(
        default=0,
        description="Total page count of the source PDF."
    )
    processed_pages_count: int = Field(
        default=0,
        description="Number of pages successfully extracted and processed."
    )
    pages: List[LucentGKPage] = Field(
        default_factory=list,
        description="List of extracted page records."
    )


# ==========================================
# 2. Legacy Question & Quiz Schemas
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
# 3. Topic Taxonomy & Matching Engine
# ==========================================

DEFAULT_GA_TOPICS = [
    "History (Ancient, Medieval, Modern India)",
    "Geography (India and World)",
    "Polity and Constitution of India",
    "Economics and Banking Awareness",
    "Science (Physics, Chemistry, Biology)",
    "Current Affairs (National and International)",
    "Awards and Honors",
    "Sports and Games",
    "Books and Authors",
    "Important Days and Events",
    "Environment and Ecology",
    "Inventions and Discoveries",
    "Cultural Heritage of India",
    "Abbreviations and Terminology"
]

TOPIC_PATTERNS: Dict[str, List[str]] = {
    "History (Ancient, Medieval, Modern India)": [
        r"\bhistory\b", r"\bindian history\b", r"\bancient india\b", r"\bmedieval india\b", r"\bmodern india\b",
        r"\bharappan\b", r"\bindus valley\b", r"\bmohenjo[- ]?daro\b", r"\bvedic\b", r"\brigveda\b",
        r"\bsamaveda\b", r"\byajurveda\b", r"\batharvaveda\b", r"\bupanishad\b", r"\bmahajanapada\b",
        r"\bmagadha\b", r"\bmaurya\b", r"\bashoka\b", r"\bchandragupta\b", r"\bgupta dynasty\b",
        r"\bgupta empire\b", r"\bharshavardhana\b", r"\bbuddhism\b", r"\bgautama buddha\b",
        r"\bjainism\b", r"\bmahavira\b", r"\bdelhi sultanate\b", r"\bslave dynasty\b", r"\bkhilji\b",
        r"\btughlaq\b", r"\bsayyid\b", r"\blodi\b", r"\bmughal\b", r"\bbabur\b", r"\bhumayun\b",
        r"\bakbar\b", r"\bjahangir\b", r"\bshah jahan\b", r"\baurangzeb\b", r"\bmaratha\b", r"\bshivaji\b",
        r"\bpeshwa\b", r"\bvijayanagar\b", r"\bbahmani\b", r"\bbhakti movement\b", r"\bsufism\b",
        r"\beast india company\b", r"\bbattle of plassey\b", r"\bbattle of buxar\b", r"\bviceroy\b",
        r"\bgovernor[- ]general\b", r"\bwarren hastings\b", r"\bcornwallis\b", r"\bdalhousie\b",
        r"\brevolt of 1857\b", r"\bindian national congress\b", r"\bswadeshi\b", r"\bnon[- ]cooperation\b",
        r"\bcivil disobedience\b", r"\bquit india\b", r"\bina\b", r"\bsubhash chandra bose\b",
        r"\bpartition of india\b", r"\bfreedom struggle\b", r"\bchola\b", r"\bpallava\b", r"\brashtrakuta\b"
    ],
    "Geography (India and World)": [
        r"\bgeography\b", r"\bphysical geography\b", r"\bworld geography\b", r"\bindian geography\b",
        r"\bsolar system\b", r"\bplanets?\b", r"\blatitudes?\b", r"\blongitudes?\b", r"\beclipse\b",
        r"\batmosphere\b", r"\btroposphere\b", r"\bstratosphere\b", r"\bmesosphere\b", r"\bionosphere\b",
        r"\blithosphere\b", r"\bhydrosphere\b", r"\bocean currents?\b", r"\btides?\b", r"\bcontinents?\b",
        r"\bvolcano\w*\b", r"\bearthquakes?\b", r"\bplate tectonics\b", r"\bcyclones?\b", r"\bmonsoon\b",
        r"\bdrainage system\b", r"\brivers? of india\b", r"\bganga\b", r"\bbrahmaputra\b", r"\bindus river\b",
        r"\bgodavari\b", r"\bkrishna river\b", r"\bnarmada\b", r"\btapi\b", r"\bcauvery\b",
        r"\bhimalayas?\b", r"\bpeninsular plateau\b", r"\bcoastal plains?\b", r"\bandaman\b", r"\bnicobar\b",
        r"\bsoils? of india\b", r"\balluvial soil\b", r"\bblack soil\b", r"\bred soil\b", r"\blaterite\b",
        r"\bagriculture in india\b", r"\bkharif\b", r"\brabi\b", r"\bzaid\b", r"\birrigation\b",
        r"\bmineral resources?\b", r"\bthermal power\b", r"\bhydroelectric\b", r"\btransport system\b",
        r"\brailways\b", r"\bhighways\b", r"\bmajor ports?\b"
    ],
    "Polity and Constitution of India": [
        r"\bpolity\b", r"\bconstitution of india\b", r"\bconstituent assembly\b", r"\bdrafting committee\b",
        r"\bpreamble\b", r"\bfundamental rights?\b", r"\bdirective principles\b", r"\bdpsp\b",
        r"\bfundamental duties?\b", r"\bpresident of india\b", r"\bvice[- ]president\b", r"\bprime minister\b",
        r"\bcouncil of ministers\b", r"\bparliament\b", r"\blok sabha\b", r"\brajya sabha\b", r"\bspeaker\b",
        r"\bmoney bill\b", r"\bsupreme court\b", r"\bhigh courts?\b", r"\bchief justice\b",
        r"\bwrit jurisdiction\b", r"\bhabeas corpus\b", r"\bjudiciary\b", r"\bgovernor\b",
        r"\bchief minister\b", r"\bstate legislature\b", r"\bpanchayati raj\b", r"\bmunicipalit\w+\b",
        r"\belection commission\b", r"\bcomptroller and auditor\b", r"\bcag\b", r"\bupsc\b",
        r"\bfinance commission\b", r"\battorney general\b", r"\bemergency provisions?\b",
        r"\bconstitutional amendments?\b", r"\barticle \d+\b", r"\bschedule \d+\b"
    ],
    "Economics and Banking Awareness": [
        r"\beconomics?\b", r"\bindian economy\b", r"\bgdp\b", r"\bgnp\b", r"\bnnp\b", r"\bnational income\b",
        r"\bper capita income\b", r"\beconomic planning\b", r"\bfive year plans?\b", r"\bniti aayog\b",
        r"\bplanning commission\b", r"\brbi\b", r"\breserve bank of india\b", r"\bmonetary policy\b",
        r"\brepo rate\b", r"\breverse repo\b", r"\bcrr\b", r"\bslr\b", r"\bbank rate\b", r"\bbanking\b",
        r"\bcommercial banks?\b", r"\bnationalisation\b", r"\bsbi\b", r"\bnabard\b", r"\bsidbi\b",
        r"\bsebi\b", r"\bnpa\b", r"\binflation\b", r"\bdeflation\b", r"\bconsumer price index\b",
        r"\bwholesale price index\b", r"\bunion budget\b", r"\bfiscal deficit\b", r"\brevenue deficit\b",
        r"\btaxation\b", r"\bgst\b", r"\bincome tax\b", r"\bbalance of payments?\b", r"\bforeign trade\b",
        r"\bfdi\b", r"\bpoverty line\b", r"\bunemployment\b"
    ],
    "Science (Physics, Chemistry, Biology)": [
        r"\bgeneral science\b", r"\bphysics\b", r"\bchemistry\b", r"\bbiology\b", r"\bmechanics\b",
        r"\bnewton'?s laws\b", r"\bgravitation\b", r"\bwork, energy and power\b", r"\bthermodynamics\b",
        r"\boptics\b", r"\blight\b", r"\blens\w*\b", r"\bmirror\w*\b", r"\belectricity\b", r"\bmagnetism\b",
        r"\bohm'?s law\b", r"\bradioactivity\b", r"\bnuclear physics\b", r"\batomic structure\b",
        r"\bperiodic table\b", r"\bchemical bonding\b", r"\bacids?, bases? and salts?\b", r"\bph scale\b",
        r"\bmetals? and non[- ]metals?\b", r"\borganic chemistry\b", r"\bhydrocarbons?\b", r"\bpolymers?\b",
        r"\bcell biology\b", r"\bmitosis\b", r"\bmeiosis\b", r"\bhuman physiology\b",
        r"\bdigestive system\b", r"\brespiratory system\b", r"\bcirculatory system\b", r"\bheart\b",
        r"\bnervous system\b", r"\bendocrine system\b", r"\bhormones?\b", r"\bgenetics\b",
        r"\bdna\b", r"\brna\b", r"\bphotosynthesis\b", r"\bplant hormones?\b", r"\bvitamins?\b",
        r"\bdeficiency diseases?\b", r"\bhuman diseases?\b", r"\bbacteria\b", r"\bviruses?\b"
    ],
    "Current Affairs (National and International)": [
        r"\bcurrent affairs\b", r"\bsummits?\b", r"\bg20\b", r"\bbrics\b", r"\basean\b",
        r"\bsco summit\b", r"\bun general assembly\b", r"\bgovernment schemes?\b",
        r"\bpradhan mantri\b", r"\bbilateral\b", r"\bmilitary exercises?\b",
        r"\binternational agreements?\b"
    ],
    "Awards and Honors": [
        r"\bawards? and honou?rs?\b", r"\bbharat ratna\b", r"\bpadma vibhushan\b", r"\bpadma bhushan\b",
        r"\bpadma shri\b", r"\bnobel prizes?\b", r"\bramon magsaysay\b", r"\bbooker prize\b",
        r"\boscars?\b", r"\bacal?demy awards?\b", r"\bjnanpith award\b", r"\bsahitya akademi\b",
        r"\bsaraswati samman\b", r"\bvyas samman\b", r"\bdadasaheb phalke\b", r"\bparam vir chakra\b",
        r"\bmaha vir chakra\b", r"\bvir chakra\b", r"\bashoka chakra\b", r"\bgallantry awards?\b",
        r"\bshanti swarup bhatnagar\b"
    ],
    "Sports and Games": [
        r"\bsports? and games?\b", r"\bolympic games?\b", r"\bolympics\b", r"\bparalympics?\b",
        r"\basian games\b", r"\bcommonwealth games\b", r"\bfifa world cup\b", r"\bcricket world cup\b",
        r"\bhockey\b", r"\bcricket\b", r"\bfootball\b", r"\bbadminton\b", r"\blawn tennis\b",
        r"\btable tennis\b", r"\bchess\b", r"\bwimbledon\b", r"\bgrand slam\b", r"\bus open\b",
        r"\bfrench open\b", r"\baustralian open\b", r"\branji trophy\b", r"\bdurand cup\b",
        r"\bsantosh trophy\b", r"\bthomas cup\b", r"\buber cup\b", r"\bsports terminolog\w+\b",
        r"\bnumber of players\b"
    ],
    "Books and Authors": [
        r"\bbooks? and authors?\b", r"\bfamous books?\b", r"\bwritten by\b", r"\bautobiograph\w+\b",
        r"\bliterary works?\b", r"\barthashastra\b", r"\bshakuntalam\b", r"\bkalidasa\b",
        r"\bmahabharata\b", r"\bramayana\b", r"\bindica\b", r"\bmudrarakshasa\b", r"\brajatarangini\b",
        r"\bain[- ]i[- ]akbari\b", r"\bbaburnama\b", r"\bgitanjali\b", r"\brabindranath tagore\b",
        r"\bdiscovery of india\b", r"\bjawaharlal nehru\b", r"\bmy experiments with truth\b",
        r"\bmahatma gandhi\b", r"\banandamath\b"
    ],
    "Important Days and Events": [
        r"\bimportant days?\b", r"\bimportant dates?\b", r"\bworld health day\b", r"\bearth day\b",
        r"\bworld environment day\b", r"\binternational yoga day\b", r"\bnational science day\b",
        r"\brepublic day\b", r"\bindependence day\b", r"\bteachers? day\b", r"\binternational women'?s day\b",
        r"\bworld aids day\b", r"\bhuman rights day\b", r"\bworld water day\b", r"\bobserved on\b",
        r"\bcelebrated on\b"
    ],
    "Environment and Ecology": [
        r"\benvironment and ecology\b", r"\becosystem\b", r"\bfood chain\b", r"\bfood web\b",
        r"\bbiodiversity\b", r"\bbiodiversity hotspots?\b", r"\bbiosphere reserves?\b",
        r"\bnational parks? in india\b", r"\bwildlife sanctuar\w+\b", r"\bramsar sites?\b",
        r"\bwetlands?\b", r"\bproject tiger\b", r"\bproject elephant\b", r"\bglobal warming\b",
        r"\bgreenhouse effect\b", r"\bgreenhouse gases?\b", r"\bozone layer depletion\b",
        r"\bacid rain\b", r"\bpollution\b", r"\bdeforestation\b", r"\bkyoto protocol\b",
        r"\bparis agreement\b", r"\bmontreal protocol\b", r"\biucn\b", r"\bred data book\b"
    ],
    "Inventions and Discoveries": [
        r"\binventions? and discoveries\b", r"\bdiscover\w+\b", r"\binvented by\b",
        r"\bdiscovered by\b", r"\binventor\b", r"\bscientific discoveries\b",
        r"\bsteam engine\b", r"\btelephone\b", r"\belectric bulb\b", r"\bthomas edison\b",
        r"\bradio\b", r"\bmarconi\b", r"\btelevision\b", r"\bpenicillin\b", r"\balexander fleming\b",
        r"\bx[- ]rays?\b", r"\broentgen\b", r"\bpolio vaccine\b", r"\belectron\b",
        r"\bproton\b", r"\bneutron\b", r"\btheory of relativity\b"
    ],
    "Cultural Heritage of India": [
        r"\bcultural heritage\b", r"\bclassical dances? of india\b", r"\bfolk dances?\b",
        r"\bbharatanatyam\b", r"\bkathak\b", r"\bkathakali\b", r"\bkuchipudi\b", r"\bodissi\b",
        r"\bmanipuri\b", r"\bmohiniyattam\b", r"\bsattriya\b", r"\bfairs? and festivals?\b",
        r"\bkumbh mela\b", r"\bunesco world heritage\b", r"\bajanta caves?\b", r"\bellora caves?\b",
        r"\bkhajuraho\b", r"\bhampi\b", r"\bkonark sun temple\b", r"\btemple architecture\b",
        r"\bmonuments?\b", r"\bmusical instruments?\b", r"\bclassical music\b", r"\bfolk arts?\b"
    ],
    "Abbreviations and Terminology": [
        r"\babbreviations? and terminolog\w*\b", r"\bcommon abbreviations?\b", r"\bacronyms?\b",
        r"\bstands for\b", r"\bfull forms?\b", r"\bexpanded as\b", r"\bterminolog\w+\b",
        r"\bisro\b", r"\bdrdo\b", r"\bbarc\b", r"\bunesco\b", r"\bunicef\b", r"\bwho\b",
        r"\bwto\b", r"\bimf\b", r"\bradar\b", r"\bsonar\b", r"\blaser\b"
    ]
}


def load_ga_topics_from_file(topic_file: str | Path = "topic.txt") -> List[str]:
    """
    Parses topic.txt and extracts all subtopics listed under 'General Awareness'.
    Falls back to DEFAULT_GA_TOPICS if topic.txt is not found.
    """
    t_path = Path(topic_file)
    if not t_path.exists():
        return DEFAULT_GA_TOPICS

    try:
        with open(t_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        ga_topics: List[str] = []
        in_ga = False
        major_subjects = {"english", "general intelligence", "quantitative aptitude"}

        for line in lines:
            lower = line.lower()
            if "general awareness" in lower:
                in_ga = True
                continue
            if in_ga:
                if lower in major_subjects:
                    break
                ga_topics.append(line)

        return ga_topics if ga_topics else DEFAULT_GA_TOPICS
    except Exception as e:
        print(f"[!] Warning reading {topic_file}: {e}. Using default General Awareness topics.")
        return DEFAULT_GA_TOPICS


def match_ga_topics(
    headings: List[str],
    text: str,
    active_topic: Optional[str] = None,
    available_topics: Optional[List[str]] = None
) -> List[str]:
    """
    Matches page headings and body text against General Awareness topics.
    Applies high weighting to heading matches and tracks active section continuity.
    """
    avail = available_topics or DEFAULT_GA_TOPICS
    combined_headings = " ".join(headings).lower()
    lower_text = text.lower()

    scores: Dict[str, float] = {}

    for topic in avail:
        patterns = TOPIC_PATTERNS.get(topic, [])
        score = 0.0

        for pat in patterns:
            # 5x weight for heading matches
            h_hits = len(re.findall(pat, combined_headings, re.IGNORECASE))
            if h_hits > 0:
                score += h_hits * 5.0

            # 1x weight for body text matches (capped per pattern)
            t_hits = len(re.findall(pat, lower_text, re.IGNORECASE))
            if t_hits > 0:
                score += min(t_hits, 6) * 1.0

        if score > 0:
            scores[topic] = score

    # Continuity bias: active chapter section bonus
    if active_topic and active_topic in avail:
        scores[active_topic] = scores.get(active_topic, 0.0) + 2.0

    if not scores:
        return [active_topic] if active_topic else [avail[0]]

    # Sort descending by score
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_score = sorted_scores[0][1]

    # Select top matches that score at least 40% of the top match
    matched = [t for t, s in sorted_scores if s >= top_score * 0.4][:3]
    return matched


# ==========================================
# 4. OCR Line Clustering & Heading Extractor
# ==========================================

def cluster_ocr_boxes_into_lines(
    boxes_with_text: List[Tuple[Tuple[float, float, float, float], str]],
    line_thresh: float = 12.0
) -> List[str]:
    """
    Clusters bounding boxes that share similar vertical coordinates into horizontal lines,
    sorted left-to-right, to reconstruct natural reading flow.
    """
    if not boxes_with_text:
        return []

    # Sort primarily top to bottom
    sorted_boxes = sorted(boxes_with_text, key=lambda b: (b[0][1], b[0][0]))
    lines: List[Dict] = []

    for box, text in sorted_boxes:
        y_center = (box[1] + box[3]) / 2.0
        placed = False
        for line in lines:
            if abs(y_center - line["y_center"]) <= line_thresh:
                line["items"].append((box[0], text))
                line["y_center"] = (line["y_center"] * (len(line["items"]) - 1) + y_center) / len(line["items"])
                placed = True
                break
        if not placed:
            lines.append({"y_center": y_center, "items": [(box[0], text)]})

    # Sort lines by y_center
    lines.sort(key=lambda l: l["y_center"])

    result_lines = []
    for line in lines:
        line["items"].sort(key=lambda it: it[0])  # sort left-to-right by x0
        line_str = " ".join([it[1] for it in line["items"]]).strip()
        if line_str:
            result_lines.append(line_str)

    return result_lines


def extract_headings_from_ocr(
    results: List[Tuple],
    median_h: float
) -> List[str]:
    """
    Identifies section, unit, and chapter headings from OCR bounding boxes based on:
    - Bounding box height exceeding median
    - All-caps section lines
    - Numbered / Roman numeral headings (e.g. '1. Harappan Civilization', 'II. Vedic Period')
    """
    headings: List[str] = []
    sentence_starters = {
        "the", "according", "this", "these", "it", "they", "in", "on", "at",
        "for", "with", "by", "from", "and", "but", "while", "whereas"
    }
    noise_keywords = ["nitin", "gktrick", "telegram", "pdf notes", "download", ".com", ".in", "medium"]

    for bbox, text, conf in results:
        text_clean = text.strip()
        if len(text_clean) < 3 or conf < 0.25:
            continue

        lower = text_clean.lower()
        if any(ign in lower for ign in noise_keywords):
            continue

        words = text_clean.split()
        if len(words) > 8:
            continue

        first_word = words[0].lower().rstrip(".,:")
        if first_word in sentence_starters and not text_clean.isupper():
            continue

        # Check box height
        h = abs(bbox[2][1] - bbox[0][1])
        is_large = h >= median_h * 1.25
        is_all_caps = text_clean.isupper() and len(words) <= 6 and not text_clean.isdigit()
        is_numbered = bool(re.match(r"^(?:[0-9]{1,2}\.|[I|V|X]+\.|\([a-z0-9]\))\s+[A-Z]", text_clean))

        if is_all_caps or is_numbered or (is_large and len(words) <= 5):
            clean_h = re.sub(r"\s+", " ", text_clean).strip(" -_~:")
            if clean_h and clean_h not in headings:
                headings.append(clean_h)

    return headings


# ==========================================
# 5. Lucent GK Page Extractor
# ==========================================

class LucentGKExtractor:
    """
    Handles page-by-page OCR extraction and topic matching for the Lucent GK book.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        output_dir: str | Path = "ocr_gk_extraction",
        topic_file: str | Path = "topic.txt",
        dpi: int = 100
    ):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # Load topics
        self.ga_topics = load_ga_topics_from_file(topic_file)

        # Initialize EasyOCR reader
        base_dir = Path(__file__).parent.resolve()
        model_dir = base_dir / ".easyocr" / "model"
        user_dir = base_dir / ".easyocr" / "user_network"
        model_dir.mkdir(parents=True, exist_ok=True)
        user_dir.mkdir(parents=True, exist_ok=True)

        _ensure_easyocr()
        use_gpu = bool(torch and torch.cuda.is_available())
        print(f"[*] Initializing EasyOCR Reader (GPU={use_gpu}, model_dir='{model_dir}')...")
        self.reader = easyocr.Reader(
            ["en"],
            gpu=use_gpu,
            model_storage_directory=str(model_dir),
            user_network_directory=str(user_dir),
            verbose=False
        )
        print("[+] EasyOCR Reader ready.")

        # Subject hierarchy tracker across sequential pages
        self.active_major_heading: Optional[str] = None
        self.active_topic: Optional[str] = None

    def extract_page(self, doc: pymupdf.Document, page_idx: int) -> LucentGKPage:
        """
        Extracts content from a single page, detects headings, and tags with corresponding GA topics.
        """
        page_num = page_idx + 1  # 1-indexed
        page = doc[page_idx]

        # 1. Check if native text is present (digital PDF fallback)
        native_text = page.get_text().strip()
        if len(native_text) > 60:
            lines = [l.strip() for l in native_text.splitlines() if l.strip()]
            headings = [
                l for l in lines[:10]
                if (l.isupper() or len(l.split()) <= 6) and len(l) > 3
            ][:4]
            full_text = "\n".join(lines)
        else:
            # 2. Perform OCR on page rendering
            pix = page.get_pixmap(dpi=self.dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)

            ocr_results = self.reader.readtext(img_np, paragraph=False)

            # Compute median text height
            heights = [abs(b[2][1] - b[0][1]) for b, t, c in ocr_results if t.strip()]
            median_h = float(np.median(heights)) if heights else 20.0

            # Detect headings
            headings = extract_headings_from_ocr(ocr_results, median_h)

            # Cluster text into coherent lines
            boxes_with_text = [
                ((b[0][0], b[0][1], b[2][0], b[2][1]), t.strip())
                for b, t, c in ocr_results if t.strip() and c >= 0.2
            ]
            clustered_lines = cluster_ocr_boxes_into_lines(boxes_with_text, line_thresh=median_h * 0.5)
            full_text = "\n".join(clustered_lines)

        # 3. Update chapter / section context
        for h in headings:
            h_upper = h.upper()
            if any(k in h_upper for k in ["HISTORY", "ANCIENT INDIA", "MEDIEVAL INDIA", "MODERN INDIA"]):
                self.active_major_heading = "Indian History"
                self.active_topic = "History (Ancient, Medieval, Modern India)"
            elif any(k in h_upper for k in ["GEOGRAPHY", "SOLAR SYSTEM", "RIVERS OF INDIA", "SOILS OF INDIA"]):
                self.active_major_heading = "Geography"
                self.active_topic = "Geography (India and World)"
            elif any(k in h_upper for k in ["POLITY", "CONSTITUTION"]):
                self.active_major_heading = "Indian Polity and Constitution"
                self.active_topic = "Polity and Constitution of India"
            elif any(k in h_upper for k in ["ECONOMY", "ECONOMICS", "BANKING"]):
                self.active_major_heading = "Indian Economy"
                self.active_topic = "Economics and Banking Awareness"
            elif any(k in h_upper for k in ["PHYSICS", "CHEMISTRY", "BIOLOGY", "SCIENCE"]):
                self.active_major_heading = "General Science"
                self.active_topic = "Science (Physics, Chemistry, Biology)"
            elif "SPORTS" in h_upper or "GAMES" in h_upper:
                self.active_major_heading = "Sports"
                self.active_topic = "Sports and Games"
            elif "AWARDS" in h_upper or "HONOURS" in h_upper or "HONORS" in h_upper:
                self.active_major_heading = "Awards and Honors"
                self.active_topic = "Awards and Honors"
            elif "DAYS" in h_upper or "DATES" in h_upper:
                self.active_major_heading = "Important Days and Events"
                self.active_topic = "Important Days and Events"
            elif "ECOLOGY" in h_upper or "ENVIRONMENT" in h_upper:
                self.active_major_heading = "Environment and Ecology"
                self.active_topic = "Environment and Ecology"
            elif "INVENTIONS" in h_upper or "DISCOVERIES" in h_upper:
                self.active_major_heading = "Inventions and Discoveries"
                self.active_topic = "Inventions and Discoveries"

        # Build relative headings list including active section
        relative_headings: List[str] = []
        if self.active_major_heading and self.active_major_heading not in headings:
            relative_headings.append(self.active_major_heading)
        relative_headings.extend(headings)

        # 4. Match topics against General Awareness
        matched_topics = match_ga_topics(
            headings=relative_headings,
            text=full_text,
            active_topic=self.active_topic,
            available_topics=self.ga_topics
        )

        return LucentGKPage(
            page_number=page_num,
            relative_headings=relative_headings,
            corresponding_topics=matched_topics,
            text=full_text
        )

    def process_all_pages(
        self,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        force: bool = False
    ) -> LucentGKBookExtraction:
        """
        Processes pages sequentially, saving each page's JSON to ocr_gk_extraction/ immediately.
        """
        doc = pymupdf.open(str(self.pdf_path))
        total_doc_pages = len(doc)

        start_idx = max(0, start_page - 1)
        end_idx = min(total_doc_pages, end_page) if end_page else total_doc_pages
        if max_pages:
            end_idx = min(end_idx, start_idx + max_pages)

        print("=" * 70)
        print(f"Starting Lucent GK Extraction: {self.pdf_path.name}")
        print(f"Total Pages in PDF: {total_doc_pages}")
        print(f"Processing Range: Pages {start_idx + 1} to {end_idx}")
        print(f"Output Directory: {self.output_dir.resolve()}")
        print("=" * 70)

        all_pages: List[LucentGKPage] = []
        master_json_path = self.output_dir / "lucent_gk_extracted.json"

        for p_idx in range(start_idx, end_idx):
            page_num = p_idx + 1
            page_json_file = self.output_dir / f"page_{page_num:04d}.json"

            # Check if page was previously processed (supports instant resume)
            if page_json_file.exists() and not force:
                try:
                    with open(page_json_file, "r", encoding="utf-8") as pf:
                        page_data = LucentGKPage.model_validate_json(pf.read())
                        all_pages.append(page_data)
                        print(f"[Page {page_num:03d}/{total_doc_pages:03d}] (Cached) Headings: {page_data.relative_headings[:2]} | Topics: {page_data.corresponding_topics}")
                        continue
                except Exception:
                    pass

            t0 = time.time()
            page_data = self.extract_page(doc, p_idx)
            elapsed = time.time() - t0

            # Save individual page JSON immediately
            with open(page_json_file, "w", encoding="utf-8") as pf:
                pf.write(page_data.model_dump_json(indent=2))

            all_pages.append(page_data)

            print(
                f"[Page {page_num:03d}/{total_doc_pages:03d}] ({elapsed:4.1f}s) "
                f"Headings: {page_data.relative_headings[:2]} | "
                f"Topics: {page_data.corresponding_topics} -> {page_json_file.name}"
            )

            # Update consolidated master JSON incrementally every 5 pages or at end
            if len(all_pages) % 5 == 0 or page_num == end_idx:
                dataset = LucentGKBookExtraction(
                    source_pdf=str(self.pdf_path),
                    total_pages=total_doc_pages,
                    processed_pages_count=len(all_pages),
                    pages=all_pages
                )
                with open(master_json_path, "w", encoding="utf-8") as mf:
                    mf.write(dataset.model_dump_json(indent=2))

        # Final dataset build
        final_dataset = LucentGKBookExtraction(
            source_pdf=str(self.pdf_path),
            total_pages=total_doc_pages,
            processed_pages_count=len(all_pages),
            pages=all_pages
        )
        with open(master_json_path, "w", encoding="utf-8") as mf:
            mf.write(final_dataset.model_dump_json(indent=2))

        print("=" * 70)
        print(f"[+] Extraction Completed! Processed {len(all_pages)} pages.")
        print(f"[+] Individual JSONs saved to: {self.output_dir.resolve()}")
        print(f"[+] Consolidated dataset saved to: {master_json_path.resolve()}")
        print("=" * 70)

        return final_dataset


# ==========================================
# 6. Legacy Question Paper Extraction Functions
# ==========================================

def _clean_text(text: str) -> str:
    """Normalize whitespace and strip unwanted non-breaking characters."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\r", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def extract_questions_from_pdf(pdf_path: str | Path) -> Quiz:
    """Legacy function to extract MCQs from test paper PDFs."""
    doc = pymupdf.open(str(pdf_path))
    full_text = "\n".join([page.get_text() for page in doc])

    q_matches = list(re.finditer(r"(?:^|\n)\s*(Q\d+[\.:].*?)(?=(?:^|\n)\s*Q\d+[\.:]|\Z)", full_text, re.DOTALL))
    extracted_questions: List[Question] = []

    for match in q_matches:
        block = match.group(1).strip()
        num_m = re.match(r"^Q(\d+)[\.:]", block)
        q_num = int(num_m.group(1)) if num_m else len(extracted_questions) + 1

        ans_m = re.search(r"\bAns[\.:]?\s*\(?([a-dA-D1-4])\)?", block, re.IGNORECASE)
        content_before_ans = block[:ans_m.start()] if ans_m else block
        ans_label = ans_m.group(1).lower() if ans_m else ""

        opts_iter = list(re.finditer(r"(?:^|\n)[^\S\r\n]*\(([a-dA-D1-4])\)[^\S\r\n]*(.*?)(?=(?:^|\n)[^\S\r\n]*\([a-dA-D1-4]\)|\Z)", content_before_ans, re.DOTALL))
        options_dict: Dict[str, str] = {}
        first_opt_start = len(content_before_ans)

        if opts_iter:
            first_opt_start = opts_iter[0].start()
            for om in opts_iter:
                options_dict[om.group(1).lower()] = _clean_text(om.group(2))

        q_prompt = content_before_ans[:first_opt_start].strip()
        q_prompt = re.sub(r"^Q\d+[\.:]\s*", "", q_prompt).strip()
        q_prompt = _clean_text(q_prompt)

        labels = ["a", "b", "c", "d"]
        options_list = [f"({lbl}) {options_dict.get(lbl, '')}".strip() for lbl in labels]
        correct_answer_text = f"({ans_label}) {options_dict.get(ans_label, '')}".strip() if ans_label in options_dict else f"({ans_label})"

        extracted_questions.append(
            Question(
                question_number=q_num,
                question=q_prompt or f"Question {q_num}",
                options=options_list,
                correct_answer=correct_answer_text
            )
        )

    return Quiz(
        source_pdf=str(pdf_path),
        exam_title=Path(pdf_path).stem,
        total_questions=len(extracted_questions),
        questions=extracted_questions
    )


# ==========================================
# 7. Main CLI Execution
# ==========================================

def find_lucent_gk_pdf(books_dir: str | Path = "books") -> Path:
    """Finds Lucent GK PDF in the books directory."""
    b_path = Path(books_dir)
    if not b_path.exists():
        raise FileNotFoundError(f"Books folder '{books_dir}' does not exist.")

    candidates = list(b_path.glob("*.pdf"))
    if not candidates:
        raise FileNotFoundError(f"No PDF files found in '{books_dir}'.")

    # Prefer Lucent GK PDF if present
    for c in candidates:
        if "lucent" in c.name.lower():
            return c

    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Extract Lucent GK book page-by-page into strict Pydantic JSON tagged with General Awareness topics."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to Lucent GK PDF (default: auto-detected in 'books/')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ocr_gk_extraction",
        help="Output directory to save page-by-page JSON files (default: 'ocr_gk_extraction')"
    )
    parser.add_argument(
        "--topic-file",
        type=str,
        default="topic.txt",
        help="Path to topic.txt containing General Awareness topics (default: 'topic.txt')"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="1-indexed starting page number (default: 1)"
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="1-indexed ending page number (default: all pages)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to process in this run (default: None)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="OCR rendering resolution in DPI (default: 100)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction of pages even if page JSON already exists"
    )

    args = parser.parse_args()

    # Determine PDF path
    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        pdf_path = find_lucent_gk_pdf("books")

    extractor = LucentGKExtractor(
        pdf_path=pdf_path,
        output_dir=args.output_dir,
        topic_file=args.topic_file,
        dpi=args.dpi
    )

    extractor.process_all_pages(
        start_page=args.start_page,
        end_page=args.end_page,
        max_pages=args.max_pages,
        force=args.force
    )


if __name__ == "__main__":
    main()
