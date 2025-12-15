from __future__ import annotations
import json
import os
import math
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, List, Optional, Set


_NON_ALNUM_RE = re.compile(r'[^0-9a-zA-Z/]')  # keep slash for some dates
_CURRENCY_RE = re.compile(r'[\$\£\€]')
_COMMA_RE = re.compile(r',')
_NUMBER_RE = re.compile(r'^[-+]?\d+(\.\d+)?$')


def extract_scalars(obj: Any) -> List[str]:
    """
    Recursively extract primitive scalar values from nested JSON-like objects.
    Returns list of string representations (raw).
    """
    scalars: List[str] = []
    if obj is None:
        return scalars
    if isinstance(obj, dict):
        for v in obj.values():
            scalars.extend(extract_scalars(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            scalars.extend(extract_scalars(item))
    elif isinstance(obj, (str, int, float, bool)):
        scalars.append(str(obj))
    else:
        scalars.append(str(obj))
    return scalars


def try_parse_date(s: str) -> Optional[str]:
    """
    Try a set of common formats and a fallback heuristic to parse dates.
    Returns ISO 'YYYY-MM-DD' or None.
    """
    s0 = s.strip()
    if not s0:
        return None
    # common formats
    formats = ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y"]
    # small cleanup
    s1 = s0.replace(".", "").replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
    for fmt in formats:
        try:
            dt = datetime.strptime(s1, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    # fallback digit heuristic: find 3 numbers (m,d,y or y,m,d)
    digits = re.findall(r'\d{1,4}', s1)
    if len(digits) >= 3:
        try:
            # if first is 4-digit assume year-first
            if len(digits[0]) == 4:
                y, m, d = digits[0], digits[1], digits[2]
            else:
                m, d, y = digits[0], digits[1], digits[2]
            y = y if len(y) == 4 else ("20" + y if len(y) == 2 else y)
            dt = datetime(year=int(y), month=int(m), day=int(d))
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


def normalize_value(raw: str) -> str:
    """
    Normalize a scalar into canonical token string:
      - date: "date:YYYY-MM-DD"
      - numeric: "num:<int or float>"
      - text:  "str:<lower-alnum tokens>"
    Returns empty string for empty/invalid inputs.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if s == "":
        return ""
    # strip currency symbols
    s = _CURRENCY_RE.sub("", s)
    # try date
    date_iso = try_parse_date(s)
    if date_iso:
        return f"date:{date_iso}"
    # remove commas used as thousands sep
    # s_nc = _COMMA_RE.sub("", s)
    # s_nc = s_nc.replace("\u00A0", " ")
    # # remove percent sign but keep the number
    # s_nc = s_nc.replace("%", "")
    # numeric?
    if _NUMBER_RE.match(s.replace(" ", "")):
        try:
            # num = float(s_nc.replace(" ", ""))
            # if num.is_integer():
            #     return f"num:{int(num)}"
            return f"num:{s}"
        except Exception:
            pass
    # fallback text: lowercase, remove punctuation, collapse spaces
    s_low = s.lower()
    s_alpha = _NON_ALNUM_RE.sub(" ", s_low)
    s_alpha = re.sub(r'\s+', ' ', s_alpha).strip()
    if not s_alpha:
        return s_low
    return f"str:{s_alpha}"


class DocumentMatcher:
    """
    Matches documents to sample descriptions.
    - Add docs (doc_id -> dict)
    - Set samples (list of dicts)
    - Call match() to get sample groupings
    """
    def __init__(self, similarity_threshold: float = 0.45, noisy_value_blacklist: Optional[Set[str]] = None):
        self.similarity_threshold = similarity_threshold
        # storage
        self.documents: Dict[str, Dict[str,Any]] = {}
        self.doc_values: Dict[str, Set[str]] = {}
        self.value_index: Dict[str, Set[str]] = defaultdict(set)  # value -> set(doc_ids)
        self.samples: List[Dict[str,Any]] = []
        self.sample_signatures: List[Set[str]] = []
        # values to ignore (very noisy tokens)
        self.noisy_values = noisy_value_blacklist if noisy_value_blacklist is not None else set(
            ["str:invoice", "str:invoice number", "str:date", "str:amount", "str:total"]
        )

    # data ingestion
    def add_document(self, doc_id: str, doc: Dict[str,Any]) -> None:
        """
        Add or update a document.
        doc_id must be unique and stable (e.g. filename or DB id)
        """
        self.documents[doc_id] = deepcopy(doc)
        # extract scalars and normalize
        scalars = extract_scalars(doc)
        vals = set()
        for s in scalars:
            nv = normalize_value(s)
            if not nv or nv in self.noisy_values:
                continue
            vals.add(nv)
        # set new values and index
        self.doc_values[doc_id] = vals
        for v in vals:
            self.value_index[v].add(doc_id)

    def add_documents_bulk(self, docs: Dict[str, Dict[str,Any]]) -> None:
        for doc_id, doc in docs.items():
            self.add_document(doc_id, doc)

    def set_samples(self, sample_list: List[Dict[str,Any]]) -> None:
        """
        Set sample descriptions; each sample is a dict with fields that identify it.
        """
        self.samples = deepcopy(sample_list)
        self.sample_signatures = []
        for s in self.samples:
            scalars = extract_scalars(s)
            sig = set()
            for sc in scalars:
                nv = normalize_value(sc)
                if nv and nv not in self.noisy_values:
                    sig.add(nv)
            self.sample_signatures.append(sig)

    def _value_weight(self, value: str) -> float:
        """
        Rarer values -> larger weight. weight = 1 / log(1 + doc_freq + 1)
        """
        df = len(self.value_index.get(value, set()))
        return 1.0 / math.log(2 + df)  # log base e, +2 to avoid div-by-zero

    def score_doc_to_sample(self, doc_id: str, sample_idx: int) -> float:
        doc_vals = self.doc_values.get(doc_id, set())
        sample_vals = self.sample_signatures[sample_idx] if 0 <= sample_idx < len(self.sample_signatures) else set()
        inter = doc_vals & sample_vals

        if not inter:
            return 0.0

        # check if inter only contains date
        contains_date = any(v.startswith("date:") for v in inter)
        if contains_date and len(inter) == 1:
            return 0.0

        return sum(self._value_weight(v) for v in inter)

    def initial_assignments(self) -> Dict[int, Set[str]]:
        """
        For each sample compute doc_ids that directly match based on score >= threshold.
        """
        assign: Dict[int, Set[str]] = {i: set() for i in range(len(self.samples))}
        for doc_id in self.documents.keys():
            for i in range(len(self.samples)):
                score = self.score_doc_to_sample(doc_id, i)
                if score >= self.similarity_threshold:
                    assign[i].add(doc_id)
        return assign

    def match(self) -> Dict[int, Dict[str, Any]]:
        """
        Run full pipeline and return mapping:
          sample_idx -> {"sample_description": ..., "doc_ids": [...], "documents": [...]}
        Documents that don't match any sample are omitted; to get unmatched, compute the set difference.
        """
        if not self.samples:
            return {}
        initial = self.initial_assignments()
        out: Dict[int, Dict[str, Any]] = {}
        for i, doc_ids in initial.items():
            if not doc_ids:
                continue
            out[i] = {
                "sample_description": self.samples[i],
                "doc_ids": sorted(list(doc_ids)),
                "documents": [self.documents[d] for d in sorted(list(doc_ids))]
            }
        return out

    def export_groupings_json(self) -> str:
        groups = self.match()
        out_list = []
        for idx, info in groups.items():
            out_list.append({
                "sample_index": idx,
                "sample_description": info["sample_description"],
                "doc_ids": info["doc_ids"],
                "documents": info["documents"]
            })
        return json.dumps(out_list, indent=2, ensure_ascii=False)


def grouping(doc_dir: str, samples_file: str, out_file: str = "groupings_out.json"):
    """
    Example runner:
      - loads documents from directory
      - loads sample descriptions array from JSON file
      - runs matcher and writes output JSON
    """
    matcher = DocumentMatcher(similarity_threshold=0.45)

    # load docs; assign stable doc ids (filename-based)
    docs: Dict[str, Dict[str,Any]] = {}
    for filename in sorted(os.listdir(doc_dir)):
        full = os.path.join(doc_dir, filename)
        if os.path.isdir(full):
            continue
        if filename.lower().endswith(".json"):
            with open(full, "r", encoding="utf-8") as f:
                doc = json.load(f)
                doc_id = filename  # you can use other id scheme
                docs[doc_id] = doc
        elif filename.lower().endswith(".jsonl"):
            # name each line doc as filename:lineno
            with open(full, "r", encoding="utf-8") as f:
                for ln, line in enumerate(f, start=1):
                    line=line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    doc_id = f"{filename}:{ln}"
                    docs[doc_id] = doc
    matcher.add_documents_bulk(docs)

    # load samples
    with open(samples_file, "r", encoding="utf-8") as f:
        sample_list = json.load(f)
    matcher.set_samples(sample_list)

    # run matching
    out_json = matcher.export_groupings_json()
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(out_json)
    print(f"Groupings written to {out_file}")
    print(out_json)


if __name__ == "__main__":
    sample_docs_dir = "./document_jsons"
    samples_json_file = "./sample.json"
    if os.path.exists(sample_docs_dir) and os.path.exists(samples_json_file):
        grouping(sample_docs_dir, samples_json_file)
    else:
        print("To run example, create `./document_jsons/` with .json files and `./sample.json` file.")
