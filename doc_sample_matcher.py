from __future__ import annotations
import json
import os
import math
import re
from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

# ----------------------------
# Regex & Normalization
# ----------------------------

_NON_ALNUM_RE = re.compile(r'[^0-9a-zA-Z/]')
_CURRENCY_RE = re.compile(r'[\$\£\€]')
_NUMBER_RE = re.compile(r'^[-+]?\d+(\.\d+)?$')


# ----------------------------
# Utility Functions
# ----------------------------

def extract_scalars(obj: Any, parent_key: str = "") -> List[Tuple[str, str]]:
    """
    Recursively extract primitive scalar values from nested JSON-like objects.
    Returns list of (key, value) tuples.
    """
    scalars: List[Tuple[str, str]] = []

    if obj is None:
        return scalars

    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{parent_key}.{k}" if parent_key else k
            scalars.extend(extract_scalars(v, full_key))
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            full_key = f"{parent_key}[{idx}]" if parent_key else str(idx)
            scalars.extend(extract_scalars(item, full_key))
    else:
        # scalar value, return with its key path
        scalars.append((parent_key, str(obj)))

    return scalars


def try_parse_date(s: str) -> Optional[str]:
    formats = [
        "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d",
        "%m-%d-%Y", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y"
    ]
    s = s.strip()
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


def normalize_value(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""

    s = _CURRENCY_RE.sub("", s)

    date = try_parse_date(s)
    if date:
        return f"date:{date}"

    if _NUMBER_RE.match(s.replace(" ", "")):
        return f"num:{s.replace(',', '')}"

    s = s.lower()
    s = _NON_ALNUM_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return f"str:{s}" if s else ""


# ----------------------------
# Token Categorization
# ----------------------------

def token_category(token: str) -> str:
    if token.startswith("date:"):
        return "date"
    if token.startswith("num:"):
        return "number"
    if token.startswith("str:"):
        val = token[4:]
        if re.fullmatch(r"[A-Z]{2}", val.upper()):
            return "state"
        if re.fullmatch(r"\d{5}(-\d{4})?", val):
            return "zip"
        return "text"
    return "other"


CATEGORY_MULTIPLIER = {
    "date": 0.0,
    "number": 0.25,
    "state": 0.1,
    "zip": 0.1,
    "text": 1.0,
    "other": 0.2
}


# ----------------------------
# Matcher Class
# ----------------------------

class DocumentMatcher:
    def __init__(self, similarity_threshold: float = 0.45):
        self.similarity_threshold = similarity_threshold
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.doc_values: Dict[str, Set[str]] = {}
        self.value_index: Dict[str, Set[str]] = defaultdict(set)
        self.samples: List[Dict[str, Any]] = []
        self.sample_signatures: List[Set[str]] = set(), []

        self.noisy_elements = {
            "shipping_method", "method", "city", "state", "zip", "origin", "salesperson", "terms", "fob_point",
            "city_state_zip", "delivery_date", "subtotal", "total", "discount", "tax_rate", "salesperson",
            "carrier_name", "shipper_id", "declared_value", "seal_number", "scac_code", "total_gross_amount",
            "total_discount", "total_paid", "shipping_handling", "fob"
        }

    # ----------------------------
    # Ingestion
    # ----------------------------

    def add_document(self, doc_id: str, doc: Dict[str, Any]) -> None:
        self.documents[doc_id] = deepcopy(doc)
        vals = set()

        for s in extract_scalars(doc):
            k = s[0].split('.')[-1]
            v = normalize_value(s[1])
            if k in self.noisy_elements:
                continue
            if v and v not in self.noisy_elements:
                vals.add(v)

        self.doc_values[doc_id] = vals
        for v in vals:
            self.value_index[v].add(doc_id)

    def add_documents_bulk(self, docs: Dict[str, Dict[str, Any]]) -> None:
        for k, v in docs.items():
            self.add_document(k, v)

    def set_samples(self, sample_list: List[Dict[str, Any]]) -> None:
        self.samples = deepcopy(sample_list)
        self.sample_signatures = []

        for s in self.samples:
            sig = set()
            for sc in extract_scalars(s):
                k = sc[0].split('.')[-1]
                v = normalize_value(sc[1])
                if k in self.noisy_elements:
                    continue
                if v and v not in self.noisy_elements:
                    sig.add(v)

            self.sample_signatures.append(sig)

    # ----------------------------
    # Scoring
    # ----------------------------

    def _idf_weight(self, token: str) -> float:
        df = len(self.value_index.get(token, set()))
        return 1.0 / math.log(2 + df)

    def _score_intersection(self, inter: Set[str]) -> float:
        if not inter:
            return 0.0

        if len(inter) == 1 and next(iter(inter)).startswith("date:"):
            return 0.0

        score = 0.0
        for t in inter:
            cat = token_category(t)
            mult = CATEGORY_MULTIPLIER.get(cat, 0.2)
            score += self._idf_weight(t) * mult

        return score

    def score_doc_to_sample(self, doc_id: str, sample_idx: int) -> float:
        inter = self.doc_values.get(doc_id, set()) & self.sample_signatures[sample_idx]
        return self._score_intersection(inter)

    def score_doc_to_doc(self, a: str, b: str) -> float:
        inter = self.doc_values[a] & self.doc_values[b]
        return self._score_intersection(inter)

    # ----------------------------
    # Transitive Expansion
    # ----------------------------

    def expand_docs(self, seeds: Set[str]) -> Set[str]:
        expanded = set(seeds)
        q = deque(seeds)

        while q:
            cur = q.popleft()
            for other in self.documents:
                if other in expanded or other == cur:
                    continue
                if self.score_doc_to_doc(cur, other) >= self.similarity_threshold:
                    expanded.add(other)
                    q.append(other)

        return expanded

    # ----------------------------
    # Matching
    # ----------------------------

    def match(self) -> Dict[int, Dict[str, Any]]:
        results = {}

        for i in range(len(self.samples)):
            seeds = {
                d for d in self.documents
                if self.score_doc_to_sample(d, i) >= self.similarity_threshold
            }

            if not seeds:
                continue

            all_docs = self.expand_docs(seeds)
            results[i] = {
                "sample_description": self.samples[i],
                "doc_ids": sorted(all_docs),
                "documents": [self.documents[d] for d in sorted(all_docs)]
            }

        return results

    def export_groupings_json(self) -> str:
        return json.dumps(
            [
                {
                    "sample_index": i,
                    **v
                }
                for i, v in self.match().items()
            ],
            indent=2,
            ensure_ascii=False
        )


# ----------------------------
# Runner
# ----------------------------

def grouping(doc_dir: str, samples_file: str, out_file: str = "groupings_out.json"):
    matcher = DocumentMatcher()

    docs = {}
    for f in os.listdir(doc_dir):
        if f.endswith(".json"):
            with open(os.path.join(doc_dir, f)) as fh:
                docs[f] = json.load(fh)

    matcher.add_documents_bulk(docs)

    with open(samples_file) as f:
        matcher.set_samples(json.load(f))

    out = matcher.export_groupings_json()
    with open(out_file, "w") as f:
        f.write(out)

    print(out)


if __name__ == "__main__":
    sample_docs_dir = "./document_jsons"
    samples_json_file = "./sample.json"
    if os.path.exists(sample_docs_dir) and os.path.exists(samples_json_file):
        grouping(sample_docs_dir, samples_json_file)
    else:
        print("To run example, create `./document_jsons/` with .json files and `./sample.json` file.")
