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

HOP_DECAY = 0.6
AMBIGUITY_MARGIN = 0.15


# ----------------------------
# Utility Functions
# ----------------------------

def extract_scalars(obj: Any, parent_key: str = "") -> List[Tuple[str, str]]:
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
        scalars.append((parent_key, str(obj)))

    return scalars


def try_parse_date(s: str) -> Optional[str]:
    formats = [
        "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d",
        "%m-%d-%Y", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s.strip(), fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


def normalize_value(raw: str) -> str:
    if not raw:
        return ""

    s = _CURRENCY_RE.sub("", str(raw).strip())

    date = try_parse_date(s)
    if date:
        return f"date:{date}"

    if _NUMBER_RE.match(s.replace(",", "")):
        return f"num:{s.replace(',', '')}"

    s = _NON_ALNUM_RE.sub(" ", s.lower())
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
        v = token[4:]
        if re.fullmatch(r"[A-Z]{2}", v.upper()):
            return "state"
        if re.fullmatch(r"\d{5}(-\d{4})?", v):
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
# Matcher
# ----------------------------

class DocumentMatcher:
    def __init__(self, similarity_threshold: float = 0.45):
        self.similarity_threshold = similarity_threshold
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.doc_values: Dict[str, Set[str]] = {}
        self.value_index: Dict[str, Set[str]] = defaultdict(set)
        self.samples: List[Dict[str, Any]] = []
        self.sample_signatures: List[Set[str]] = []

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

        for k, v in extract_scalars(doc):
            if k.split(".")[-1] in self.noisy_elements:
                continue
            nv = normalize_value(v)
            if nv:
                vals.add(nv)

        self.doc_values[doc_id] = vals
        for v in vals:
            self.value_index[v].add(doc_id)

    def add_documents_bulk(self, docs: Dict[str, Dict[str, Any]]) -> None:
        for k, v in docs.items():
            self.add_document(k, v)

    def set_samples(self, samples: List[Dict[str, Any]]) -> None:
        self.samples = deepcopy(samples)
        self.sample_signatures = []

        for s in samples:
            sig = set()
            for k, v in extract_scalars(s):
                if k.split(".")[-1] in self.noisy_elements:
                    continue
                nv = normalize_value(v)
                if nv:
                    sig.add(nv)
            self.sample_signatures.append(sig)

    # ----------------------------
    # Scoring
    # ----------------------------

    def _idf(self, t: str) -> float:
        return 1.0 / math.log(2 + len(self.value_index.get(t, [])))

    def _score_intersection(self, inter: Set[str]) -> float:
        if not inter:
            return 0.0
        if len(inter) == 1 and next(iter(inter)).startswith("date:"):
            return 0.0

        return sum(
            self._idf(t) * CATEGORY_MULTIPLIER[token_category(t)]
            for t in inter
        )

    def score_doc_to_sample(self, d: str, i: int) -> float:
        return self._score_intersection(
            self.doc_values[d] & self.sample_signatures[i]
        )

    def score_doc_to_doc(self, a: str, b: str) -> float:
        return self._score_intersection(
            self.doc_values[a] & self.doc_values[b]
        )

    # ----------------------------
    # Graph traversal (hop aware)
    # ----------------------------

    def _best_path_score(self, start: str, sample_idx: int) -> float:
        q = deque([(start, 0, 1.0)])
        seen = {start}

        best = 0.0
        while q:
            cur, hops, conf = q.popleft()

            direct = self.score_doc_to_sample(cur, sample_idx)
            if direct >= self.similarity_threshold:
                best = max(best, conf * direct)

            for other in self.documents:
                if other in seen or other == cur:
                    continue
                s = self.score_doc_to_doc(cur, other)
                if s >= self.similarity_threshold:
                    seen.add(other)
                    q.append((other, hops + 1, conf * HOP_DECAY))

        return best

    # ----------------------------
    # Matching (single assignment)
    # ----------------------------

    def match(self) -> Dict[int, Dict[str, Any]]:
        doc_assignment: Dict[str, Tuple[int, float]] = {}

        for doc in self.documents:
            scores = {
                i: self._best_path_score(doc, i)
                for i in range(len(self.samples))
            }

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if not ranked or ranked[0][1] == 0:
                continue

            if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < AMBIGUITY_MARGIN:
                continue  # ambiguous → unassigned

            doc_assignment[doc] = ranked[0]

        results: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            "doc_ids": [],
            "documents": []
        })

        for d, (i, _) in doc_assignment.items():
            results[i]["doc_ids"].append(d)
            results[i]["documents"].append(self.documents[d])
            results[i]["sample_description"] = self.samples[i]

        for r in results.values():
            r["doc_ids"].sort()

        return results

    def export_groupings_json(self) -> str:
        return json.dumps(
            [{"sample_index": i, **v} for i, v in self.match().items()],
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


# a = b = c = d = e

# a => sample_1
# e => sample_2
#
#
#
#
# We have doc_1 which ==> Sample_1
# We have doc_2 which ==> Sample_2
# We have doc_3 which ==> doc_1 & doc_2
