"""
Topic Classifier & Dataset Enricher with Class Imbalance Mitigation
===================================================================
Uses scikit-learn (TfidfVectorizer + LinearSVC) to classify exam questions
into Subjects and Subtopics defined in `topic.txt`.

Implements advanced class imbalance mitigation techniques:
1. SMOTE (Synthetic Minority Over-sampling Technique) in TF-IDF feature space:
   - Interpolates between k-nearest neighbors of rare classes (e.g. 'Cloze test', 'Venn diagrams', 'LCM and HCF').
   - Ensures minority topics have adequate representation in the feature space.
2. Cost-Sensitive Loss Weighting (`class_weight='balanced'` in LinearSVC):
   - Automatically weights the loss function inversely proportional to class frequencies.
   - Prevents majority classes (e.g. 'Fill in the Blanks', 'Percentage') from dominating the decision boundary.
3. Hierarchical Classification Architecture:
   - Level 1: Subject Classifier (English, General Intelligence, Quant, General Awareness).
   - Level 2: Subtopic Classifier (constrained within the candidate subtopics of predicted subject).
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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
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
# 2. Refined High-Precision Rule Engine
# ==============================================================

def rule_based_subtopic(q_text: str, options: List[str]) -> Optional[str]:
    """
    Extracts high-confidence subtopic using distinct SSC exam patterns.
    Prioritizes specific minority categories (e.g. 'Cloze test') before generic ones ('Fill in the blanks').
    """
    opts_combined = " ".join(options)
    text = f"{q_text} {opts_combined}".lower()

    # --- 1. ENGLISH ---
    # Prioritize Cloze test before general fill in the blanks
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
    if re.search(r"\b(correctly spelt|incorrectly spelt|spelling error|spelt word|misspelt|\bspelling\b)", text):
        return "Spellings"
    if re.search(r"\b(antonym|synonym|opposite in meaning|similar in meaning)\b", text):
        return "Synonyms and antonyms"
    if re.search(r"\b(grammatical error|contains a grammatical error|contains an error|spotting error|divided into four segments|split into four segments|segment that contains)\b", text):
        return "Error detection (spotting and correcting errors)"
    if re.search(r"\b(rearrange the following|replace the highlighted|substitute the underlined|underlined segment|highlighted part|meaningful and grammatically correct paragraph)\b", text):
        return "Sentence improvement and rearrangement"
    if re.search(r"\b(fill in the blank|blank no|fill in blank|blank no\.)\b", text):
        return "Fill in the blanks"
    if re.search(r"\b(passage carefully|according to the passage|based on the passage|in the passage)\b", text):
        return "Reading comprehension"
    if re.search(r"\b(uses the word .* with the same meaning|closest meaning|contextual meaning)\b", text):
        return "Vocabulary-based questions"
    if re.search(r"\b(complete the sentence|sentence completion)\b", text):
        return "Sentence completion"

    # --- 2. GENERAL INTELLIGENCE ---
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

    # --- 3. QUANTITATIVE APTITUDE ---
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

    # --- 4. GENERAL AWARENESS ---
    if re.search(r"\b(biodiversity|greenhouse gas|ramsar site|wildlife sanctuary|ecosystem|global warming|pollution|ozone|endangered species|wetland|biosphere reserve)\b", text):
        return "Environment and Ecology"
    if re.search(r"\b(article \d+|constitution of india|fundamental rights|fundamental duties|president of india|supreme court|parliament|lok sabha|rajya sabha|preamble|governor|constitutional amendment|high court|panchayati raj|election commission|attorney general)\b", text):
        return "Polity and Constitution of India"
    if re.search(r"\b(dynasty|empire|emperor|mughal|battle of|ashoka|harappan|indus valley|sultanate|revolt of 1857|viceroy|mauryan|gupta|chola|delhi sultanate|british raj|akbar|babur|aurangzeb|freedom movement)\b", text):
        return "History (Ancient, Medieval, Modern India)"
    if re.search(r"\b(river|mountain|plateau|tributary|national park|hemisphere|monsoon|gulf|soil type|strait|peninsula|western ghats|eastern ghats|himalayas|equator|tropic of)\b", text):
        return "Geography (India and World)"
    if re.search(r"\b(acid|base|cell|hormone|vitamin|photosynthesis|periodic table|newton|kinetic energy|atomic number|chemical formula|bacteria|virus|disease|human body|enzyme|chlorophyll|element|compound)\b", text):
        return "Science (Physics, Chemistry, Biology)"
    if re.search(r"\b(gdp|inflation|rbi|fiscal deficit|repo rate|monetary policy|five-year plan|niti aayog|revenue|goods and services tax|gst|budget|national income)\b", text):
        return "Economics and Banking Awareness"
    if re.search(r"\b(olympic|cricket|trophy|fifa|badminton|chess|tennis|world cup|medal|stadium|khel ratna|asian games|commonwealth games|player|bowler|batsman)\b", text):
        return "Sports and Games"
    if re.search(r"\b(nobel prize|bharat ratna|padma shri|padma bhushan|padma vibhushan|sahitya akademi|oscar|grammy|pulitzer|gallantry award)\b", text):
        return "Awards and Honors"
    if re.search(r"\b(folk dance|classical dance|temple|festival|gharana|carnatic|hindustani|monument|unesco|musical instrument|raga|sitar|tabla)\b", text):
        return "Cultural Heritage of India"
    if re.search(r"\b(autobiography|written by|author of the book|novel|who wrote)\b", text):
        return "Books and Authors"
    if re.search(r"\b(celebrated on|observed on|world environment day|national youth day|international day)\b", text):
        return "Important Days and Events"
    if re.search(r"\b(invented by|discovered by|invention|discovery)\b", text):
        return "Inventions and Discoveries"
    if re.search(r"\b(stands for|full form of|abbreviation|acronym)\b", text):
        return "Abbreviations and Terminology"
    if re.search(r"\b(summit|isro|launched in \d{4}|mission|g20|appointed as|cabinet minister)\b", text):
        return "Current Affairs (National and International)"

    return None


# ==============================================================
# 3. SMOTE (Synthetic Minority Over-sampling Technique)
# ==============================================================

def smote_oversample_tfidf(
    X: sparse.csr_matrix,
    y: List[str] | np.ndarray,
    target_min: int = 15,
    k_neighbors: int = 3,
    random_state: int = 42
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Performs SMOTE oversampling directly in sparse TF-IDF feature space.
    For any minority class with fewer than `target_min` samples:
    - If samples >= 2: interpolates between k-nearest neighbors along feature lines.
    - If samples == 1: synthesizes non-zero feature jitter perturbations.
    Re-normalizes all generated feature vectors to unit Euclidean (L2) norm.
    """
    rng = np.random.default_rng(random_state)
    y_arr = np.array(y)
    classes, counts = np.unique(y_arr, return_counts=True)

    new_X = [X]
    new_y = list(y_arr)
    oversampled_classes = 0
    synthetic_samples_added = 0

    for cls, count in zip(classes, counts):
        if count < target_min:
            needed = target_min - count
            idx = np.where(y_arr == cls)[0]
            cls_X = X[idx].toarray() if sparse.issparse(X) else X[idx]
            oversampled_classes += 1
            synthetic_samples_added += needed

            if count == 1:
                # 1-sample class: synthetic feature jitter on non-zero components
                base_vec = cls_X[0]
                non_zero = np.where(base_vec > 0)[0]
                for _ in range(needed):
                    synthetic = base_vec.copy()
                    if len(non_zero) > 0:
                        # Randomly jitter weights of active vocabulary terms
                        noise_factors = rng.uniform(0.7, 1.3, size=len(non_zero))
                        synthetic[non_zero] *= noise_factors
                    norm = np.linalg.norm(synthetic)
                    if norm > 0:
                        synthetic /= norm
                    new_X.append(sparse.csr_matrix(synthetic.reshape(1, -1)))
                    new_y.append(cls)
            else:
                # Multi-sample minority class: SMOTE k-NN convex interpolation
                k = min(k_neighbors, count - 1)
                nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(cls_X)
                _, neighbors = nn.kneighbors(cls_X)

                for _ in range(needed):
                    base_idx = rng.integers(0, count)
                    neighbor_candidates = neighbors[base_idx][1:]
                    if len(neighbor_candidates) == 0:
                        neighbor_idx = base_idx
                    else:
                        neighbor_idx = rng.choice(neighbor_candidates)

                    lam = rng.uniform(0.15, 0.85)
                    synthetic = cls_X[base_idx] + lam * (cls_X[neighbor_idx] - cls_X[base_idx])
                    norm = np.linalg.norm(synthetic)
                    if norm > 0:
                        synthetic /= norm
                    new_X.append(sparse.csr_matrix(synthetic.reshape(1, -1)))
                    new_y.append(cls)

    X_resampled = sparse.vstack(new_X)
    y_resampled = np.array(new_y)
    return X_resampled, y_resampled


# ==============================================================
# 4. Balanced Hierarchical Classifier
# ==============================================================

class BalancedTopicClassifier:
    """
    Two-stage hierarchical classifier combining:
    1. TF-IDF feature extraction (unigrams + bigrams)
    2. SMOTE oversampling for minority categories
    3. LinearSVC with loss-function class-weighting (`class_weight='balanced'`)
    """
    def __init__(
        self,
        topic_map: Dict[str, List[str]],
        subtopic_to_subject: Dict[str, str],
        use_smote: bool = True,
        target_min_samples: int = 15,
        class_weight: str = "balanced"
    ):
        self.topic_map = topic_map
        self.subtopic_to_subject = subtopic_to_subject
        self.use_smote = use_smote
        self.target_min_samples = target_min_samples
        self.class_weight = class_weight

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True,
            stop_words="english"
        )
        self.clf_subject = LinearSVC(
            C=1.0,
            max_iter=3000,
            class_weight=self.class_weight,
            random_state=42
        )
        self.subtopic_models: Dict[str, Any] = {}
        self.is_fitted = False

    def train(self, texts: List[str], subtopics: List[str], verbose: bool = True):
        """Fits TF-IDF, applies SMOTE and cost-sensitive loss weighting to train the model."""
        subjects = [self.subtopic_to_subject[st] for st in subtopics]

        # 1. Fit Global TF-IDF
        X_vec = self.vectorizer.fit_transform(texts)

        # 2. Train Subject Classifier with balanced class weights
        if self.use_smote:
            X_subj_res, y_subj_res = smote_oversample_tfidf(
                X_vec, subjects, target_min=max(self.target_min_samples, 100), random_state=42
            )
            self.clf_subject.fit(X_subj_res, y_subj_res)
        else:
            self.clf_subject.fit(X_vec, subjects)

        # 3. Train per-subject Subtopic Classifiers
        if verbose:
            print("\n[SMOTE & Class-Weighting Report per Subject]")

        for subj, valid_subtopics in self.topic_map.items():
            subj_indices = [i for i, s in enumerate(subjects) if s == subj]
            if not subj_indices:
                continue

            X_subj = X_vec[subj_indices]
            y_subj_subt = [subtopics[i] for i in subj_indices]

            distinct_classes = set(y_subj_subt)
            if len(distinct_classes) <= 1:
                self.subtopic_models[subj] = distinct_classes.pop()
                continue

            orig_counts = Counter(y_subj_subt)
            rare_classes = [c for c, cnt in orig_counts.items() if cnt < self.target_min_samples]

            if self.use_smote:
                X_sub_res, y_sub_res = smote_oversample_tfidf(
                    X_subj, y_subj_subt, target_min=self.target_min_samples, random_state=42
                )
                if verbose:
                    print(f"  • {subj:<22}: {len(y_subj_subt)} -> {len(y_sub_res)} samples "
                          f"({len(rare_classes)} minority classes boosted to min {self.target_min_samples})")
            else:
                X_sub_res, y_sub_res = X_subj, np.array(y_subj_subt)
                if verbose:
                    print(f"  • {subj:<22}: {len(y_subj_subt)} samples (class_weight='{self.class_weight}')")

            clf_sub = LinearSVC(
                C=1.0,
                max_iter=3000,
                class_weight=self.class_weight,
                random_state=42
            )
            clf_sub.fit(X_sub_res, y_sub_res)
            self.subtopic_models[subj] = clf_sub

        self.is_fitted = True

    def predict(self, text: str, subject_hint: Optional[str] = None) -> Tuple[str, str]:
        """
        Predicts (subject, subtopic) for a question text.
        Guarantees that the subtopic strictly belongs to the predicted subject according to `topic.txt`.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been trained yet.")

        vec = self.vectorizer.transform([text])

        # Step 1: Predict Subject
        if subject_hint and subject_hint in self.topic_map:
            subject = subject_hint
        else:
            subject = str(self.clf_subject.predict(vec)[0])

        # Step 2: Predict Subtopic within Subject
        sub_model = self.subtopic_models.get(subject)
        if isinstance(sub_model, str):
            subtopic = sub_model
        elif sub_model is not None:
            subtopic = str(sub_model.predict(vec)[0])
        else:
            subtopic = self.topic_map[subject][0]

        return subject, subtopic


# ==============================================================
# 5. Pipeline Execution & JSON Dataset Enrichment
# ==============================================================

def _resolve_data_path(primary: str | Path, alt_dir: str = "client/dist/assets") -> Path:
    p = Path(primary)
    if p.exists():
        return p
    alt = Path(alt_dir) / primary
    if alt.exists():
        return alt
    return p


def analyze_and_enrich_dataset(
    json_path: str | Path = "client/dist/assets/extracted_json/all_extracted_questions.json",
    individual_dir: str | Path = "client/dist/assets/extracted_json",
    topic_file: str | Path = "topic.txt",
    use_smote: bool = True,
    target_min: int = 15,
    class_weight: str = "balanced"
) -> Dict[str, Any]:
    """
    Loads dataset, trains balanced classifier with SMOTE and class-weighting,
    enriches every question with topic tags, and updates the JSON files.
    """
    topic_map, subtopic_to_subject = load_topic_hierarchy(topic_file)

    master_path = _resolve_data_path(json_path)
    if not master_path.exists():
        raise FileNotFoundError(f"Dataset JSON file not found: {master_path.resolve()}")

    with open(master_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Gather questions and seed labels
    X_train_texts = []
    y_train_subtopics = []
    all_questions_flat = []

    for paper in dataset.get("papers", []):
        for q in paper.get("questions", []):
            q_text = q.get("question", "")
            opts = q.get("options", [])
            combined_text = f"{q_text} {' '.join(opts)}"
            all_questions_flat.append((paper, q, combined_text))

            subt = rule_based_subtopic(q_text, opts)
            if subt and subt in subtopic_to_subject:
                X_train_texts.append(combined_text)
                y_train_subtopics.append(subt)

    print("\n" + "=" * 75)
    print("STEP 1: TOPIC TAXONOMY & DATASET INSPECTION")
    print("=" * 75)
    print(f"Total Papers in Dataset      : {len(dataset.get('papers', []))}")
    print(f"Total Questions in Dataset   : {len(all_questions_flat)}")
    print(f"Labeled Seed Examples        : {len(X_train_texts)} ({len(X_train_texts)/len(all_questions_flat)*100:.1f}%)")

    # Train Balanced Classifier
    print("\n" + "=" * 75)
    print("STEP 2: TRAINING WITH SMOTE OVERSAMPLING & LOSS CLASS WEIGHTS")
    print("=" * 75)
    print(f"Techniques Applied:")
    print(f"  [x] SMOTE Feature-Space Oversampling (target_min={target_min} per minority class)")
    print(f"  [x] Cost-Sensitive Loss Weighting (LinearSVC class_weight='{class_weight}')")

    classifier = BalancedTopicClassifier(
        topic_map=topic_map,
        subtopic_to_subject=subtopic_to_subject,
        use_smote=use_smote,
        target_min_samples=target_min,
        class_weight=class_weight
    )

    # Evaluation on Holdout Set (80/20 train/test split)
    X_tr, X_te, y_tr, y_te = train_test_split(X_train_texts, y_train_subtopics, test_size=0.2, random_state=42)
    eval_clf = BalancedTopicClassifier(
        topic_map=topic_map,
        subtopic_to_subject=subtopic_to_subject,
        use_smote=use_smote,
        target_min_samples=target_min,
        class_weight=class_weight
    )
    eval_clf.train(X_tr, y_tr, verbose=False)
    y_preds = [eval_clf.predict(t)[1] for t in X_te]
    holdout_acc = accuracy_score(y_te, y_preds)
    print(f"\nHoldout Test Accuracy (Balanced Model): {holdout_acc:.4f} (across 50+ classes)")

    # Full training
    classifier.train(X_train_texts, y_train_subtopics, verbose=True)
    print("\n[+] Balanced Model training completed successfully!")

    # Step 3: Classify every question in the dataset
    print("\n" + "=" * 75)
    print("STEP 3: ENRICHING QUESTIONS & UPDATING JSON DATASET")
    print("=" * 75)

    subject_counts: Counter = Counter()
    subtopic_counts: Dict[str, Counter] = defaultdict(Counter)

    for paper in dataset.get("papers", []):
        for q in paper.get("questions", []):
            q_text = q.get("question", "")
            opts = q.get("options", [])
            combined_text = f"{q_text} {' '.join(opts)}"

            rule_subt = rule_based_subtopic(q_text, opts)
            if rule_subt and rule_subt in subtopic_to_subject:
                final_subt = rule_subt
                final_subj = subtopic_to_subject[rule_subt]
            else:
                final_subj, final_subt = classifier.predict(combined_text)

            q["subject"] = final_subj
            q["subtopic"] = final_subt
            q["topic"] = final_subt

            subject_counts[final_subj] += 1
            subtopic_counts[final_subj][final_subt] += 1

    # Save master JSON
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"[+] Updated Master Dataset: {master_path.resolve()}")

    # Update individual per-PDF JSON files
    indiv_dir = Path(individual_dir)
    updated_files = 0
    if indiv_dir.exists():
        for paper in dataset.get("papers", []):
            source_stem = Path(paper.get("source_pdf", "")).stem
            indiv_file = indiv_dir / f"{source_stem}.json"
            if indiv_file.exists():
                with open(indiv_file, "w", encoding="utf-8") as f:
                    json.dump(paper, f, indent=2, ensure_ascii=False)
                updated_files += 1
        print(f"[+] Updated {updated_files} individual paper JSON files in: {indiv_dir.resolve()}")

    # Print Breakdown Report
    print("\n" + "=" * 75)
    print("ANALYSIS SUMMARY: BALANCED QUESTIONS GROUPED BY SUBJECT & SUBTOPIC")
    print("=" * 75)
    total_q = sum(subject_counts.values())

    for subj in ["English", "General Intelligence", "Quantitative Aptitude", "General Awareness"]:
        count = subject_counts[subj]
        pct = (count / total_q * 100) if total_q else 0
        print(f"\n[{subj.upper()}] - Total: {count} questions ({pct:.1f}%)")
        print("-" * 65)
        for subt, sub_c in subtopic_counts[subj].most_common():
            sub_pct = sub_c / count * 100
            print(f"  • {subt:<45}: {sub_c:3d} ({sub_pct:4.1f}%)")

    print("\n" + "=" * 75)
    print(f"Successfully processed and tagged {total_q} questions across 4 subjects!")
    print("=" * 75)

    return {
        "total_questions": total_q,
        "subject_counts": dict(subject_counts),
        "subtopic_counts": {k: dict(v) for k, v in subtopic_counts.items()}
    }


# ==============================================================
# 6. CLI Execution
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify exam questions using SMOTE and LinearSVC with balanced loss weights."
    )
    default_json = str(_resolve_data_path("client/dist/assets/extracted_json/all_extracted_questions.json"))
    default_dir = str(_resolve_data_path("client/dist/assets/extracted_json"))

    parser.add_argument(
        "--dataset",
        type=str,
        default=default_json,
        help="Path to all_extracted_questions.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_dir,
        help="Directory containing individual JSON files to update"
    )
    parser.add_argument(
        "--topics",
        type=str,
        default="topic.txt",
        help="Path to topic.txt"
    )
    parser.add_argument(
        "--target-min",
        type=int,
        default=15,
        help="Minimum target sample count for SMOTE oversampling (default: 15)"
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling (use class weights only)"
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Test query string to predict subject and subtopic for"
    )

    args = parser.parse_args()

    if args.predict:
        topic_map, subtopic_to_subject = load_topic_hierarchy(args.topics)
        with open(args.dataset, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        X_train_texts, y_train_subtopics = [], []
        for paper in dataset.get("papers", []):
            for q in paper.get("questions", []):
                subt = rule_based_subtopic(q.get("question", ""), q.get("options", []))
                if subt and subt in subtopic_to_subject:
                    X_train_texts.append(f"{q.get('question', '')} {' '.join(q.get('options', []))}")
                    y_train_subtopics.append(subt)

        clf = BalancedTopicClassifier(
            topic_map=topic_map,
            subtopic_to_subject=subtopic_to_subject,
            use_smote=not args.no_smote,
            target_min_samples=args.target_min,
            class_weight="balanced"
        )
        clf.train(X_train_texts, y_train_subtopics, verbose=False)
        subj, subt = clf.predict(args.predict)
        print(f"\nQuery   : {args.predict}")
        print(f"Subject : {subj}")
        print(f"Subtopic: {subt}\n")
    else:
        analyze_and_enrich_dataset(
            json_path=args.dataset,
            individual_dir=args.output_dir,
            topic_file=args.topics,
            use_smote=not args.no_smote,
            target_min=args.target_min,
            class_weight="balanced"
        )
