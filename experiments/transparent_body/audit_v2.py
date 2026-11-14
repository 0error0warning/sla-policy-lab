"""Read only the saved v2 archive; summaries are descriptive, not model scores."""

import argparse
import json
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

EXPECTED = "07276f3173123600cf2ba4506f88ea2d7e24a0842133a5653f3c461ca9d6670f"


def audit(source):
    digest = sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED:
        raise ValueError("not the saved #25 v2 archive")
    rows = []
    with ZipFile(source) as z:
        for name in sorted(z.namelist()):
            if "/records/" not in name or not name.endswith(".json"):
                continue
            r = json.loads(z.read(name))
            if not r["thinking"]:
                continue
            s = r["output_text"]
            words = re.findall(r"\w+|[^\w\s]", s.lower())
            grams = Counter(
                tuple(words[i : i + 12]) for i in range(max(0, len(words) - 11))
            )
            repeated = sum(v - 1 for v in grams.values())
            rows.append(
                {
                    "model": r["model"],
                    "case": r["id"],
                    "member": name,
                    "raw_sha256": sha256(z.read(name)).hexdigest(),
                    "generated_tokens": r["generated_tokens"],
                    "finish": r["finish_reason"],
                    "final_present": r["final_text"] is not None,
                    "close_markers": s.count("</think>"),
                    "message_end_markers": s.count("<|im_end|>"),
                    "duplicate_12gram_fraction": repeated / max(1, sum(grams.values())),
                    "format_word_count": len(
                        re.findall(
                            r"\b(?:json|decimal|markdown|keys)\b", s, re.IGNORECASE
                        )
                    ),
                    "recalculation_word_count": len(
                        re.findall(
                            r"re-?calculat\w*|re-?check\w*|double[- ]check",
                            s,
                            re.IGNORECASE,
                        )
                    ),
                    "tail": s[-600:],
                }
            )
    if len(rows) != 8:
        raise ValueError("expected eight v2 thinking outputs")
    return {
        "source_sha256": digest,
        "records": rows,
        "caveat": "word/n-gram counts do not classify every token as unproductive; no final answer was recovered or rescored",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(audit(args.source), ensure_ascii=False, indent=2) + "\n",
        encoding="utf8",
    )
