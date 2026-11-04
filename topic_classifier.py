"""
Topic Classifier & Dataset Enricher with Targeted Oversampling & Variance Balancing
===================================================================================
Uses scikit-learn (TfidfVectorizer + LinearSVC) to classify exam questions
into Subjects and Subtopics defined in `topic.txt`.

Key Features:
1. Equal Subject Distribution (25 Questions per Category):
   - Enforces the natural 100-question exam structure: exactly 25 questions each
     for English, General Intelligence, Quantitative Aptitude, and General Awareness.
2. Targeted Oversampling Strategy:
   - Specifically targets underrepresented subtopics in the English and General Awareness
     sections using SMOTE feature-space synthesis.
   - Boosts minority classes (e.g. Cloze Test, Voice, Speech, Environment, Books & Authors,
     Important Days, Inventions, Awards) to higher training sample quotas.
3. In-Section Variance Balancing:
   - Configures adaptive soft-cap variance damping within each 25-question section
     to balance internal subtopic variance without altering the overall 25-question class size.
4. Cost-Sensitive Loss Weighting:
   - Applies `class_weight='balanced'` in LinearSVC to penalize errors on rare classes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC


# ==============================================================
# 1. Topic Taxonomy Loader
# ==============================================================

def load_topic_hierarchy(topic_file: str | Path = "topic.txt") -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Parses `topic.txt` into:
    - topic_map: {subject: [subtopic1, subtopic2, ...]}
    - subtopic_to_subject: {subtopic: subject}
    """
    topic_path = Path(topic_file)
    if not topic_path.exists():
        raise FileNotFoundError(f"Topic file not found: {topic_path.resolve()}")

    with open(topic_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    known_subjects = [
        "English",
        "General Intelligence",
        "Quantitative Aptitude",
        "General Awareness"
    ]

    topic_map: Dict[str, List[str]] = {}
    subtopic_to_subject: Dict[str, str] = {}
    current_subject: Optional[str] = None

    for line in lines:
        if line in known_subjects:
            current_subject = line
            topic_map[current_subject] = []
        elif current_subject:
            topic_map[current_subject].append(line)
            subtopic_to_subject[line] = current_subject

    return topic_map, subtopic_to_subject


# ==============================================================
# 2. Targeted Oversampling & Variance Balancing Configuration
# ==============================================================

# Target sample minimums for specific minority subtopics during SMOTE training
TARGETED_SMOTE_QUOTAS: Dict[str, int] = {
    # English targeted subtopics
    "Cloze test": 90,
    "Active and passive voice": 80,
    "Direct and indirect speech": 70,
    "One-word substitution": 70,
    "Spellings": 80,
    "Idioms and phrases": 80,
    "Reading comprehension": 60,
    "Sentence completion": 40,
    "Vocabulary-based questions": 40,
    "Comprehension-based grammar questions": 30,

    # General Awareness targeted subtopics
    "Polity and Constitution of India": 90,
    "History (Ancient, Medieval, Modern India)": 90,
    "Geography (India and World)": 90,
    "Sports and Games": 80,
    "Cultural Heritage of India": 80,
    "Current Affairs (National and International)": 70,
    "Books and Authors": 60,
    "Environment and Ecology": 60,
    "Awards and Honors": 50,
    "Important Days and Events": 50,
    "Inventions and Discoveries": 50,
    "Abbreviations and Terminology": 40,
}

# Maximum recommended questions per subtopic within a single 25-question section
# to prevent majority topics from monopolizing the section and balance internal variance
SECTION_SUBTOPIC_CAPS: Dict[str, Dict[str, int]] = {
    "English": {
        "Cloze test": 5,
        "Reading comprehension": 5,
        "Synonyms and antonyms": 4,
        "Fill in the blanks": 4,
        "Sentence improvement and rearrangement": 4,
        "Error detection (spotting and correcting errors)": 3,
        "Spellings": 3,
        "Idioms and phrases": 3,
        "Active and passive voice": 2,
        "Direct and indirect speech": 2,
        "One-word substitution": 2,
        "Vocabulary-based questions": 2,
        "Sentence completion": 2,
        "Comprehension-based grammar questions": 2,
    },
    "General Awareness": {
        "History (Ancient, Medieval, Modern India)": 4,
        "Polity and Constitution of India": 4,
        "Geography (India and World)": 4,
        "Science (Physics, Chemistry, Biology)": 4,
        "Economics and Banking Awareness": 4,
        "Cultural Heritage of India": 4,
        "Current Affairs (National and International)": 3,
        "Sports and Games": 3,
        "Books and Authors": 2,
        "Environment and Ecology": 2,
        "Awards and Honors": 2,
        "Important Days and Events": 2,
        "Inventions and Discoveries": 2,
        "Abbreviations and Terminology": 2,
    }
}


# ==============================================================
# 3. High-Precision Domain Pattern Matching
# ==============================================================

def match_domain_subtopic(q_text: str, options: List[str], expected_subject: Optional[str] = None) -> Optional[str]:
    """
    Extracts high-confidence subtopic using distinct pattern matching,
    optionally guided by the expected subject section.
    """
    opts_combined = " ".join(options)
    text = f"{q_text} {opts_combined}".lower()

    # --- 1. ENGLISH PATTERNS ---
    if expected_subject is None or expected_subject == "English":
        if re.search(r"\b(words have been deleted|words have been omitted|cloze test)\b", text) or (
            re.search(r"\bblank no\b|\bblank no\.", text) and ("passage" in text or "select the most appropriate option to fill" in text)
        ):
            return "Cloze test"
        if re.search(r"\b(passive voice|active voice|voice transformation|active form|passive form)\b", text):
            return "Active and passive voice"
        if re.search(r"\b(indirect speech|direct speech|reported speech|direct voice|indirect voice|indirect form|direct form)\b", text):
            return "Direct and indirect speech"
        if re.search(r"\b(one-word substitute|one word substitution|one-word substitution|substitute for:)\b", text):
            return "One-word substitution"
        if re.search(r"\b(meaning of the idiom|idiom:|idiomatic expression|\bidiom\b)", text) and "same meaning as" not in text:
            return "Idioms and phrases"
        if re.search(r"\b(correctly spelt|incorrectly spelt|spelling error|spelt word|misspelt|\bspelling\b|correct use of spellings?)\b", text):
            return "Spellings"
        if re.search(r"\b(antonym|synonym|opposite in meaning|similar in meaning)\b", text):
            return "Synonyms and antonyms"
        if re.search(r"\b(grammatical error|contains a grammatical error|contains an error|spotting error|divided into four segments|split into four segments|segment that contains|divided into parts|grammatically correct sentence)\b", text):
            return "Error detection (spotting and correcting errors)"
        if re.search(r"\b(rearrange the following|replace the highlighted|substitute the underlined|underlined segment|highlighted part|meaningful and grammatically correct paragraph|jumbled order|arrange the sentences|arrange the parts)\b", text):
            return "Sentence improvement and rearrangement"
        if re.search(r"\b(fill in the blank|blank no|fill in blank|blank no\.)\b", text):
            return "Fill in the blanks"
        if re.search(r"\b(passage carefully|according to the passage|based on the passage|in the passage)\b", text):
            return "Reading comprehension"
        if re.search(r"\b(uses the word .* with the same meaning|closest meaning|contextual meaning)\b", text):
            return "Vocabulary-based questions"
        if re.search(r"\b(complete the sentence|sentence completion)\b", text):
            return "Sentence completion"

    # --- 2. GENERAL INTELLIGENCE PATTERNS ---
    if expected_subject is None or expected_subject == "General Intelligence":
        if re.search(r"\b(venn diagram|diagram best represents the relationship|diagram best represents|relationship among)\b", text):
            return "Venn diagrams"
        if re.search(r"\b(code language|coded as|in a certain code)\b", text) or (re.search(r"\bcode\b", text) and any(w in text for w in ["ka, li", "written as", "is coded"])):
            return "Coding and decoding"
        if re.search(r"\b(mirror image|water image|mirror is placed)\b", text):
            return "Mirror and water images"
        if re.search(r"\b(folded twice|paper is folded|unfolded|paper cutting)\b", text):
            return "Paper folding and paper cutting"
        if re.search(r"\b(hidden \(rotation not allowed\)|figure is hidden|embedded figure|rotation is not allowed|rotation not allowed)\b", text):
            return "Embedded figures"
        if re.search(r"\b(opposite faces|dice|two positions of a dice)\b", text) or (re.search(r"\bcube\b", text) and "cuboid" not in text and "volume" not in text and "edge" not in text):
            return "Cube and dice problems"
        if re.search(r"\b(blood relation|sister of|brother of|mother of|father of|pointing to a photograph|how is .* related to|maternal|paternal)\b", text):
            return "Blood relations"
        if re.search(r"\b(walks .* metres|facing north|facing south|turns to his left|turns to her right|south-west|north-east|starting point)\b", text):
            return "Directions and distances"
        if re.search(r"\b(statements:.*conclusions:|all .* are .*|some .* are .*|no .* is .*|conclusions:)\b", text):
            return "Syllogism"
        if re.search(r"\b(statement and conclusion|conclusions?\b)", text) and "statements" in text:
            return "Statements and conclusions"
        if re.search(r"\b(statement and assumption|assumptions?\b)", text) and "statements" in text:
            return "Statement and assumptions"
        if re.search(r"\b(statement and argument|arguments?\b)", text) and "statements" in text:
            return "Statement and arguments"
        if re.search(r"\b(odd one out|odd pair|odd letter-cluster|word that is different|select the odd|different from the others)\b", text):
            return "Odd one out"
        if re.search(r"\b(interchange of signs|interchanged to make|signs should be interchanged|\* signs|correct mathematical signs)\b", text):
            return "Mathematical operations and reasoning"
        if re.search(r"\b(complete the series|missing number in the series|wrong number in the series|series:|\bseries\b|find the next number|find the next term)", text):
            return "Series"
        if re.search(r"\b[A-Za-z0-9]+\s*:\s*[A-Za-z0-9]+\s*::", q_text) or re.search(r"\b(related in the same way|related to the third|select the set in which|\banalogy\b)", text):
            return "Analogies"
        if re.search(r"\b(clock|calendar|day of the week|leap year)\b", text):
            return "Clocks and calendars"
        if re.search(r"\b(data sufficiency|seating arrangement|circular table|linear row)\b", text):
            return "Data sufficiency and logical puzzles"

    # --- 3. QUANTITATIVE APTITUDE PATTERNS ---
    if expected_subject is None or expected_subject == "Quantitative Aptitude":
        if re.search(r"\b(lcm|hcf|greatest common divisor|least common multiple|highest common factor)\b", text):
            return "LCM and HCF"
        if re.search(r"\b(probability|standard deviation|variance|mode|median)\b", text):
            return "Probability and Statistics"
        if re.search(r"\b(sin\b|cos\b|tan\b|cot\b|sec\b|cosec\b|sin\^|cos\^|tan\^|elevation|depression|trigonometric|trigonometry)\b", text):
            return "Trigonometry (ratios, heights and distances)"
        if re.search(r"\b(triangle|circle|chord|tangent|radius|diameter|quadrilateral|hypotenuse|bisector|incenter|circumcenter|centroid)\b", text):
            return "Geometry (triangles, circles, quadrilaterals)"
        if re.search(r"\b(surface area|volume|cylinder|cone|sphere|hemisphere|cuboid|perimeter|rhombus|trapezium|area of the)\b", text):
            return "Mensuration (area, volume, perimeter)"
        if re.search(r"\b(train|speed of|km/h|m/s|relative speed|boat|stream|upstream|downstream|average speed)\b", text):
            return "Time, Speed, and Distance"
        if re.search(r"\b(can complete a work in|days to complete the work|working together|efficiency|can do a piece of work)\b", text):
            return "Time and Work"
        if re.search(r"\b(pipe|cistern|leak|pipes a and b|emptied in)\b", text):
            return "Pipes and Cisterns"
        if re.search(r"\b(marked price|discount of|successive discounts|single discount)\b", text):
            return "Discount"
        if re.search(r"\b(cost price|selling price|profit of|loss of|gain of|loss percentage|profit percentage)\b", text):
            return "Profit and Loss"
        if re.search(r"\b(compound interest|simple interest|invested at|sum of money amounts to|p\.a\.|rate of interest)\b", text):
            return "Simple and Compound Interest"
        if re.search(r"\b(partnership|invested capital|shares of a and b)\b", text):
            return "Partnership"
        if re.search(r"\b(percentage|percent|% of|increased by \d+%|decreased by \d+%)\b", text):
            return "Percentage"
        if re.search(r"\b(ratio of|proportion|fourth proportional|third proportional|mean proportional|divided among .* in the ratio)\b", text):
            return "Ratio and Proportion"
        if re.search(r"\b(average of|average weight|average age|average score|average marks)\b", text):
            return "Average"
        if re.search(r"\b(x \+ 1/x|factorization|polynomial|quadratic equation|x\^2|x\^3)\b", text):
            return "Algebra (basic equations, factorization)"
        if re.search(r"\b(divisible by|remainder when|unit digit of|prime number|rational number)\b", text):
            return "Number system"
        if re.search(r"\b(bar graph|pie chart|histogram|study the given table|given table represents)\b", text):
            return "Data Interpretation (tables, charts, graphs)"
        if re.search(r"\b(arithmetic progression|geometric progression|\bap\b|\bgp\b)\b", text):
            return "Sequence and Series"
        if re.search(r"\b(simplify|evaluate|value of the expression|value of\b|how much .* remains)\b", text):
            return "Simplification and Approximation"

    # --- 4. GENERAL AWARENESS PATTERNS ---
    if expected_subject is None or expected_subject == "General Awareness":
        if re.search(r"\b(biodiversity|greenhouse gas|ramsar site|wildlife sanctuary|ecosystem|global warming|pollution|ozone|endangered species|wetland|biosphere reserve|iucn|food chain|food web|trophic level|ecological)\b", text):
            return "Environment and Ecology"
        if re.search(r"\b(article \d+|constitution|fundamental rights?|fundamental duties?|president of india|supreme court|parliament|lok sabha|rajya sabha|preamble|governor|amendment|high court|panchayati raj|election commission|attorney general|cag|comptroller|prime minister|ordinance|bill|schedule|union list|state list|concurrent list|dpsp|directive principles|writs?|habeas corpus|mandamus|judiciary|legislative|citizenship|impeachment)\b", text):
            return "Polity and Constitution of India"
        if re.search(r"\b(dynasty|empire|emperor|mughal|battle|ashoka|harappan|indus valley|sultanate|revolt|viceroy|mauryan|gupta|chola|delhi sultanate|british raj|akbar|babur|aurangzeb|freedom movement|congress|satyagraha|treaty|reign|ruler|inscriptions?|edicts?|panipat|plassey|buxar|swaraj|khilafat|sepoy|vedic|aryans?|maratha|peshwa)\b", text):
            return "History (Ancient, Medieval, Modern India)"
        if re.search(r"\b(river|mountain|plateau|tributary|national park|hemisphere|monsoon|gulf|soil|strait|peninsula|western ghats|eastern ghats|himalayas|equator|tropic of|latitude|longitude|coast|port|lake|waterfall|island|pass|dam|canal|rainfall|delta|estuary|cyclone|ocean|sea|desert|thar)\b", text):
            return "Geography (India and World)"
        if re.search(r"\b(acid|base|cell|hormone|vitamin|photosynthesis|periodic table|newton|kinetic energy|atomic number|chemical formula|bacteria|virus|disease|human body|enzyme|chlorophyll|element|compound|optics|lens|mirror|light|sound|wave|velocity|acceleration|gravity|force|work|power|joule|watt|pascal|dna|rna|protein|blood|heart|kidney|liver|neuron|skeleton|bone|metal|non-metal|electron|proton|neutron|isotope|valency|solution|catalyst|combustion|respiration)\b", text):
            return "Science (Physics, Chemistry, Biology)"
        if re.search(r"\b(gdp|inflation|rbi|fiscal deficit|repo rate|monetary policy|five-year plan|niti aayog|revenue|goods and services tax|gst|budget|national income|banking|bank|currency|sebi|stock|tax|taxation|subsidy|poverty|unemployment|microeconomics|macroeconomics|crr|slr|demand|supply)\b", text):
            return "Economics and Banking Awareness"
        if re.search(r"\b(olympic|cricket|trophy|fifa|badminton|chess|tennis|world cup|medal|stadium|khel ratna|asian games|commonwealth games|player|bowler|batsman|hockey|football|wrestling|boxing|archery|shooting|swimming|grand slam|wimbledon|ranji)\b", text):
            return "Sports and Games"
        if re.search(r"\b(nobel prize|bharat ratna|padma shri|padma bhushan|padma vibhushan|sahitya akademi|oscar|grammy|pulitzer|gallantry award|param vir chakra|dadasaheb phalke|arjuna award|dronacharya)\b", text):
            return "Awards and Honors"
        if re.search(r"\b(folk dance|classical dance|temple|festival|gharana|carnatic|hindustani|monument|unesco|musical instrument|raga|sitar|tabla|bharatanatyam|kathak|kathakali|kuchipudi|odissi|mohiniyattam|sattriya|manipuri|chhau|bihu|garba|lavani|ghoomar|yakshagana|khajuraho|ajanta|ellora|fair|mela|puja)\b", text):
            return "Cultural Heritage of India"
        if re.search(r"\b(autobiography|written by|author of the book|novel|who wrote|memoir|biography|book titled)\b", text):
            return "Books and Authors"
        if re.search(r"\b(celebrated on|observed on|world environment day|national youth day|international day|world health day|earth day|science day|constitution day)\b", text):
            return "Important Days and Events"
        if re.search(r"\b(invented by|discovered by|invention|discovery|inventor)\b", text):
            return "Inventions and Discoveries"
        if re.search(r"\b(stands for|full form of|abbreviation|acronym)\b", text):
            return "Abbreviations and Terminology"
        if re.search(r"\b(summit|isro|launched in \d{4}|mission|g20|appointed as|cabinet minister|bilateral exercise|joint exercise|cop28|brics|asean)\b", text):
            return "Current Affairs (National and International)"

    return None


# ==============================================================
# 4. Targeted SMOTE Oversampling
# ==============================================================

def targeted_smote_oversample(
    X: sparse.csr_matrix,
    y: List[str] | np.ndarray,
    baseline_min: int = 15,
    targeted_quotas: Optional[Dict[str, int]] = None,
    k_neighbors: int = 3,
    random_state: int = 42
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Applies targeted SMOTE feature synthesis in sparse TF-IDF space.
    Targeted classes receive custom oversampling thresholds to ensure balanced representation.
    """
    rng = np.random.default_rng(random_state)
    y_arr = np.array(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    quotas = targeted_quotas or {}

    new_X = [X]
    new_y = list(y_arr)

    for cls, count in zip(classes, counts):
        target_count = quotas.get(cls, baseline_min)
        if count < target_count:
            needed = target_count - count
            idx = np.where(y_arr == cls)[0]
            cls_X = X[idx].toarray() if sparse.issparse(X) else X[idx]

            if count == 1:
                base_vec = cls_X[0]
                non_zero = np.where(base_vec > 0)[0]
                for _ in range(needed):
                    synthetic = base_vec.copy()
                    if len(non_zero) > 0:
                        noise = rng.uniform(0.75, 1.25, size=len(non_zero))
                        synthetic[non_zero] *= noise
                    norm = np.linalg.norm(synthetic)
                    if norm > 0:
                        synthetic /= norm
                    new_X.append(sparse.csr_matrix(synthetic.reshape(1, -1)))
                    new_y.append(cls)
            else:
                k = min(k_neighbors, count - 1)
                nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(cls_X)
                _, neighbors = nn.kneighbors(cls_X)

                for _ in range(needed):
                    base_idx = rng.integers(0, count)
                    cand = neighbors[base_idx][1:]
                    neighbor_idx = rng.choice(cand) if len(cand) > 0 else base_idx
                    lam = rng.uniform(0.15, 0.85)
                    synthetic = cls_X[base_idx] + lam * (cls_X[neighbor_idx] - cls_X[base_idx])
                    norm = np.linalg.norm(synthetic)
                    if norm > 0:
                        synthetic /= norm
                    new_X.append(sparse.csr_matrix(synthetic.reshape(1, -1)))
                    new_y.append(cls)

    return sparse.vstack(new_X), np.array(new_y)


# ==============================================================
# 5. Targeted Topic Classifier
# ==============================================================

class TargetedTopicClassifier:
    """
    Classifier implementing:
    - Subject-bound subtopic modeling (English, GI, Quant, GA)
    - Targeted SMOTE oversampling for minority categories
    - Cost-sensitive loss weighting (`class_weight='balanced'`)
    - In-section variance balancing within the 25-question limit
    """
    def __init__(
        self,
        topic_map: Dict[str, List[str]],
        subtopic_to_subject: Dict[str, str],
        use_smote: bool = True,
        baseline_min: int = 15,
        targeted_quotas: Optional[Dict[str, int]] = None,
        class_weight: str = "balanced"
    ):
        self.topic_map = topic_map
        self.subtopic_to_subject = subtopic_to_subject
        self.use_smote = use_smote
        self.baseline_min = baseline_min
        self.targeted_quotas = targeted_quotas or TARGETED_SMOTE_QUOTAS
        self.class_weight = class_weight

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=16000,
            sublinear_tf=True,
            stop_words="english"
        )
        self.subtopic_models: Dict[str, Any] = {}
        self.is_fitted = False

    def train(self, texts: List[str], subtopics: List[str], verbose: bool = True):
        """Fits TF-IDF and trains per-subject LinearSVC models with targeted oversampling."""
        subjects = [self.subtopic_to_subject[st] for st in subtopics]
        X_vec = self.vectorizer.fit_transform(texts)

        if verbose:
            print("\n[Targeted Oversampling & Class-Weighting Report]")

        for subj, valid_subtopics in self.topic_map.items():
            subj_indices = [i for i, s in enumerate(subjects) if s == subj]
            if not subj_indices:
                continue

            X_subj = X_vec[subj_indices]
            y_subj_subt = [subtopics[i] for i in subj_indices]

            distinct = set(y_subj_subt)
            if len(distinct) <= 1:
                self.subtopic_models[subj] = distinct.pop()
                continue

            orig_len = len(y_subj_subt)
            if self.use_smote:
                X_res, y_res = targeted_smote_oversample(
                    X_subj,
                    y_subj_subt,
                    baseline_min=self.baseline_min,
                    targeted_quotas=self.targeted_quotas,
                    random_state=42
                )
                if verbose:
                    targeted_hits = [c for c in set(y_res) if c in self.targeted_quotas]
                    print(f"  • {subj:<24}: {orig_len} -> {len(y_res)} samples "
                          f"({len(targeted_hits)} targeted subtopics oversampled)")
            else:
                X_res, y_res = X_subj, np.array(y_subj_subt)

            clf = LinearSVC(
                C=1.0,
                max_iter=3500,
                class_weight=self.class_weight,
                random_state=42
            )
            clf.fit(X_res, y_res)
            self.subtopic_models[subj] = clf

        self.is_fitted = True

    def predict_single(self, text: str, subject: str) -> str:
        """Predicts subtopic for a single question strictly within the specified subject."""
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been trained.")
        if subject not in self.topic_map:
            raise ValueError(f"Unknown subject: {subject}")

        sub_model = self.subtopic_models.get(subject)
        if isinstance(sub_model, str):
            return sub_model
        if sub_model is None:
            return self.topic_map[subject][0]

        vec = self.vectorizer.transform([text])
        return str(sub_model.predict(vec)[0])

    def predict_section_balanced(
        self,
        questions: List[Dict[str, Any]],
        subject: str,
        balance_variance: bool = True
    ) -> List[str]:
        """
        Classifies all questions within a section (e.g. 25 questions) bound to `subject`.
        Applies soft-cap variance balancing for English and General Awareness so internal
        subtopic variance is minimized without altering the 25-question section size.
        """
        if not questions:
            return []

        model = self.subtopic_models.get(subject)
        if isinstance(model, str):
            return [model] * len(questions)

        caps = SECTION_SUBTOPIC_CAPS.get(subject, {})
        valid_topics = self.topic_map[subject]
        assigned_subtopics: List[Optional[str]] = [None] * len(questions)
        assigned_counts: Counter = Counter()

        # Step 1: Assign high-confidence domain pattern matches
        unassigned_indices = []
        for i, q in enumerate(questions):
            q_text = q.get("question", "")
            opts = q.get("options", [])
            rule_subt = match_domain_subtopic(q_text, opts, expected_subject=subject)

            # Accept rule match if it belongs to this subject
            if rule_subt and rule_subt in valid_topics:
                assigned_subtopics[i] = rule_subt
                assigned_counts[rule_subt] += 1
            else:
                unassigned_indices.append(i)

        if not unassigned_indices or model is None:
            # Fallback any remaining unassigned
            for idx in unassigned_indices:
                assigned_subtopics[idx] = valid_topics[0]
            return [st or valid_topics[0] for st in assigned_subtopics]

        # Step 2: Vectorize unassigned questions and compute decision scores
        unassigned_texts = [
            f"{questions[idx].get('question', '')} {' '.join(questions[idx].get('options', []))}"
            for idx in unassigned_indices
        ]
        X_unassigned = self.vectorizer.transform(unassigned_texts)

        # Decision scores for each candidate subtopic
        raw_scores = model.decision_function(X_unassigned)
        decision_scores = np.atleast_2d(raw_scores)
        classes = list(model.classes_)

        # Step 3: Variance-balanced assignment using soft caps
        for row_i, orig_idx in enumerate(unassigned_indices):
            scores = decision_scores[row_i].copy()

            # If binary classification (shape (1,)):
            if len(scores) == 1 and len(classes) == 2:
                chosen_topic = classes[0] if scores[0] < 0 else classes[1]
                assigned_subtopics[orig_idx] = chosen_topic
                assigned_counts[chosen_topic] += 1
                continue

            # Apply variance damping penalty if topic has reached or exceeded soft cap
            adjusted_scores = scores.copy()
            if balance_variance and caps:
                for col_j, cls_name in enumerate(classes):
                    if col_j < len(adjusted_scores):
                        cap = caps.get(cls_name, 3)
                        curr_count = assigned_counts[cls_name]
                        if curr_count >= cap:
                            adjusted_scores[col_j] -= 0.65 * (curr_count - cap + 1)

            best_j = int(np.argmax(adjusted_scores))
            chosen_topic = classes[best_j]
            assigned_subtopics[orig_idx] = chosen_topic
            assigned_counts[chosen_topic] += 1

        return [st or valid_topics[0] for st in assigned_subtopics]


# ==============================================================
# 6. Pipeline Execution & JSON Dataset Enrichment
# ==============================================================

def analyze_and_enrich_dataset(
    json_path: str | Path = "extracted_json/all_extracted_questions.json",
    individual_dir: str | Path = "extracted_json",
    topic_file: str | Path = "topic.txt",
    use_smote: bool = True,
    baseline_min: int = 15,
    class_weight: str = "balanced"
) -> Dict[str, Any]:
    """
    1. Loads dataset of 100-question exam papers.
    2. Maps each 25-question section to its respective category (English, GI, Quant, GA).
    3. Trains the classifier with targeted oversampling for English and GA subtopics.
    4. Applies in-section variance balancing to distribute subtopics evenly within each 25-question section.
    5. Saves updated JSON files and outputs analytical summary.
    """
    topic_map, subtopic_to_subject = load_topic_hierarchy(topic_file)

    master_path = Path(json_path)
    if not master_path.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {master_path.resolve()}")

    with open(master_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Step 1: Collect training data bound to section ground truth
    X_train_texts = []
    y_train_subtopics = []
    total_q_count = 0

    for paper in dataset.get("papers", []):
        qs = paper.get("questions", [])
        n = len(qs)
        total_q_count += n
        q1 = n // 4
        q2 = 2 * q1
        q3 = 3 * q1

        sec_slices = [
            (0, q1, "English"),
            (q1, q2, "General Intelligence"),
            (q2, q3, "Quantitative Aptitude"),
            (q3, n, "General Awareness")
        ]

        for start, end, subj in sec_slices:
            for q in qs[start:end]:
                q_text = q.get("question", "")
                opts = q.get("options", [])
                combined = f"{q_text} {' '.join(opts)}"

                subt = match_domain_subtopic(q_text, opts, expected_subject=subj)
                if subt and subt in subtopic_to_subject and subtopic_to_subject[subt] == subj:
                    X_train_texts.append(combined)
                    y_train_subtopics.append(subt)

    print("\n" + "=" * 80)
    print("STEP 1: TOPIC TAXONOMY & EQUAL CATEGORY STRUCTURE (25 QUESTIONS PER CATEGORY)")
    print("=" * 80)
    print(f"Total Papers in Dataset       : {len(dataset.get('papers', []))}")
    print(f"Total Questions in Dataset    : {total_q_count}")
    print(f"Section Distribution Strategy : Exactly 25 Questions per Category per 100-Question Paper")
    print(f"High-Confidence Seed Training : {len(X_train_texts)} examples")

    # Step 2: Train Targeted Model
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING MODEL WITH TARGETED OVERSAMPLING & CLASS WEIGHTS")
    print("=" * 80)
    print("Targeted Oversampling Config:")
    print("  [x] English: Targeted quotas on Cloze test, Voice, Speech, Spellings, Idioms, Substitutions")
    print("  [x] General Awareness: Targeted quotas on History, Polity, Geography, Environment, Books, Sports")
    print("  [x] Loss Function: Cost-sensitive hinge loss with class_weight='balanced'")

    classifier = TargetedTopicClassifier(
        topic_map=topic_map,
        subtopic_to_subject=subtopic_to_subject,
        use_smote=use_smote,
        baseline_min=baseline_min,
        targeted_quotas=TARGETED_SMOTE_QUOTAS,
        class_weight=class_weight
    )

    classifier.train(X_train_texts, y_train_subtopics, verbose=True)

    # Step 3: Classify and balance questions within each 25-question section
    print("\n" + "=" * 80)
    print("STEP 3: ENRICHING DATASET & BALANCING INTERNAL VARIANCE (25 Qs / SECTION)")
    print("=" * 80)

    subject_counts: Counter = Counter()
    subtopic_counts: Dict[str, Counter] = defaultdict(Counter)

    for paper in dataset.get("papers", []):
        qs = paper.get("questions", [])
        n = len(qs)
        q1 = n // 4
        q2 = 2 * q1
        q3 = 3 * q1

        sec_slices = [
            (0, q1, "English"),
            (q1, q2, "General Intelligence"),
            (q2, q3, "Quantitative Aptitude"),
            (q3, n, "General Awareness")
        ]

        for start, end, subj in sec_slices:
            sec_questions = qs[start:end]
            # Apply variance-balanced assignment for English and GA
            is_target_sec = subj in ["English", "General Awareness"]
            assigned_topics = classifier.predict_section_balanced(
                sec_questions,
                subject=subj,
                balance_variance=is_target_sec
            )

            for q, subt in zip(sec_questions, assigned_topics):
                q["subject"] = subj
                q["subtopic"] = subt
                q["topic"] = subt
                subject_counts[subj] += 1
                subtopic_counts[subj][subt] += 1

    # Save Master JSON
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"[+] Updated Master Dataset: {master_path.resolve()}")

    # Update individual per-PDF JSON files
    indiv_dir = Path(individual_dir)
    updated_files = 0
    if indiv_dir.exists():
        for paper in dataset.get("papers", []):
            stem = Path(paper.get("source_pdf", "")).stem
            indiv_file = indiv_dir / f"{stem}.json"
            if indiv_file.exists():
                with open(indiv_file, "w", encoding="utf-8") as f:
                    json.dump(paper, f, indent=2, ensure_ascii=False)
                updated_files += 1
        print(f"[+] Updated {updated_files} individual paper JSON files in: {indiv_dir.resolve()}")

    # Print Breakdown Report
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY: 25 QUESTIONS PER CATEGORY & BALANCED INTERNAL VARIANCE")
    print("=" * 80)
    total_q = sum(subject_counts.values())

    for subj in ["English", "General Intelligence", "Quantitative Aptitude", "General Awareness"]:
        count = subject_counts[subj]
        pct = (count / total_q * 100) if total_q else 0
        avg_per_paper = count / len(dataset.get("papers", [])) if dataset.get("papers") else 0
        print(f"\n[{subj.upper()}] - Total: {count} questions ({pct:.2f}% | Avg: {avg_per_paper:.1f} Qs/paper)")
        print("-" * 70)
        for subt, sub_c in subtopic_counts[subj].most_common():
            sub_pct = sub_c / count * 100
            per_paper = sub_c / len(dataset.get("papers", [])) if dataset.get("papers") else 0
            print(f"  • {subt:<46}: {sub_c:4d} ({sub_pct:4.1f}% | ~{per_paper:.1f} per paper)")

    print("\n" + "=" * 80)
    print(f"Successfully processed and tagged {total_q} questions across all 4 equal categories!")
    print("=" * 80)

    return {
        "total_questions": total_q,
        "subject_counts": dict(subject_counts),
        "subtopic_counts": {k: dict(v) for k, v in subtopic_counts.items()}
    }


# ==============================================================
# 7. CLI Execution
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify exam questions with targeted oversampling and 25-question category balancing."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="extracted_json/all_extracted_questions.json",
        help="Path to all_extracted_questions.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extracted_json",
        help="Directory containing individual JSON files to update"
    )
    parser.add_argument(
        "--topics",
        type=str,
        default="topic.txt",
        help="Path to topic.txt"
    )
    parser.add_argument(
        "--baseline-min",
        type=int,
        default=15,
        help="Baseline minimum samples for SMOTE (default: 15)"
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling"
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Test query string to predict topic for"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="English",
        choices=["English", "General Intelligence", "Quantitative Aptitude", "General Awareness"],
        help="Subject section context for prediction (default: 'English')"
    )

    args = parser.parse_args()

    if args.predict:
        topic_map, subtopic_to_subject = load_topic_hierarchy(args.topics)
        with open(args.dataset, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        X_train_texts, y_train_subtopics = [], []
        for paper in dataset.get("papers", []):
            qs = paper.get("questions", [])
            n = len(qs)
            q1, q2, q3 = n // 4, 2 * (n // 4), 3 * (n // 4)
            sec_slices = [(0, q1, "English"), (q1, q2, "General Intelligence"), (q2, q3, "Quantitative Aptitude"), (q3, n, "General Awareness")]
            for start, end, subj in sec_slices:
                for q in qs[start:end]:
                    subt = match_domain_subtopic(q.get("question", ""), q.get("options", []), expected_subject=subj)
                    if subt and subtopic_to_subject[subt] == subj:
                        X_train_texts.append(f"{q.get('question', '')} {' '.join(q.get('options', []))}")
                        y_train_subtopics.append(subt)

        clf = TargetedTopicClassifier(
            topic_map=topic_map,
            subtopic_to_subject=subtopic_to_subject,
            use_smote=not args.no_smote,
            baseline_min=args.baseline_min,
            targeted_quotas=TARGETED_SMOTE_QUOTAS,
            class_weight="balanced"
        )
        clf.train(X_train_texts, y_train_subtopics, verbose=False)
        subt = clf.predict_single(args.predict, subject=args.subject)
        print(f"\nQuery   : {args.predict}")
        print(f"Subject : {args.subject}")
        print(f"Subtopic: {subt}\n")
    else:
        analyze_and_enrich_dataset(
            json_path=args.dataset,
            individual_dir=args.output_dir,
            topic_file=args.topics,
            use_smote=not args.no_smote,
            baseline_min=args.baseline_min,
            class_weight="balanced"
        )
