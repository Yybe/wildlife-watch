"""Re-measure cross-split byte-identical leakage in the YOLO detection corpus.

Why: the detector's headline mAP depends on it. `notebooks/05` reports 40 byte-identical groups and 17
spanning more than one split, but mAP is computed on the *test* split, so the number that matters is how many
test images also appear in train or valid. This script answers that separately and writes it in a form
`scripts/verify_claims.py` can check, rather than leaving it as prose.

Run:  .venv/Scripts/python.exe DL_PBL/scripts/audit_yolo_split_overlap.py
"""
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TREE = ROOT / "DL_PBL/data/raw/_yolo/final_data"
OUT = ROOT / "DL_PBL/outputs/logs/yolo_split_overlap.txt"
SPLITS = ("train", "valid", "test")


def main() -> int:
    if not TREE.is_dir():
        print(f"FAIL no detection tree at {TREE}")
        return 1
    by_hash = defaultdict(list)
    counts = {}
    for split in SPLITS:
        imgs = sorted((TREE / split / "images").glob("*"))
        counts[split] = len(imgs)
        for p in imgs:
            by_hash[hashlib.md5(p.read_bytes()).hexdigest()].append(split)
    dupes = {h: s for h, s in by_hash.items() if len(s) > 1}
    multi = {h: s for h, s in dupes.items() if len(set(s)) > 1}
    pairs = Counter("+".join(sorted(set(s))) for s in multi.values())
    # A test image that also exists in train (or valid) is the leak that inflates mAP. Every copy in such a
    # group counts, so a group holding two test copies plus a train copy contributes 2, not 1.
    leaked_test = sum(s.count("test") for s in multi.values()
                      if "test" in s and any(x != "test" for x in s))

    lines = [f"images_{s}={counts[s]}" for s in SPLITS]
    lines += [
        f"groups_byte_identical={len(dupes)}",
        f"extra_copies={sum(len(s) - 1 for s in dupes.values())}",
        f"groups_spanning_splits={len(multi)}",
        f"pairs_test_train={pairs.get('test+train', 0)}",
        f"pairs_train_valid={pairs.get('train+valid', 0)}",
        f"pairs_test_valid={pairs.get('test+valid', 0)}",
        f"test_images_duplicated_elsewhere={leaked_test}",
        f"test_share_duplicated={round(leaked_test / max(counts['test'], 1), 4)}",
        "note=only byte-identical copies are counted; near-duplicates (average-hash hamming<=4) are reported "
        "separately in notebooks/05 and were not filtered",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    assert sum(len(s) - 1 for s in dupes.values()) == 41, "drop count changed from the notebook's audit"
    assert len(multi) == 17, "cross-split group count changed from the notebook's audit"
    print("consistency with notebooks/05: 41 extra copies and 17 cross-split groups reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
