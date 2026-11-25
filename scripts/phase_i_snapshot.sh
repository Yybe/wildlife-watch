#!/usr/bin/env bash
# Freeze the exact evaluated version into Phase_I/<timestamp>/ with a manifest of what is inside.
# The hackathon instructions require a timestamped Phase_I (and Phase_II) folder, and "reproducible
# README", so this records hashes of the files being submitted rather than just copying them.
#
# Usage: bash DL_PBL/scripts/phase_i_snapshot.sh [phase_i|phase_ii]
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
ROOT="$(pwd)"
PY="$ROOT/.venv/Scripts/python.exe"
PHASE="${1:-phase_i}"
STAMP="$(date +%Y%m%dT%H%M%S)"
DEST="$ROOT/DL_PBL/$([ "$PHASE" = "phase_i" ] && echo Phase_I || echo Phase_II)/$STAMP"
mkdir -p "$DEST" || exit 1

for item in README.md PLAN.md requirements.txt docs src notebooks tests configs scripts data/splits; do
  # `cp -r DL_PBL/data/splits DEST/` would land at DEST/splits, breaking the paths the docs cite, so the
  # parent of a nested item is created first to preserve the relative layout.
  case "$item" in */*) mkdir -p "$DEST/$(dirname "$item")"; parent="$DEST/$(dirname "$item")";;
        *) parent="$DEST";; esac
  [ -e "DL_PBL/$item" ] && cp -r "DL_PBL/$item" "$parent/" || echo "missing $item (skipped)"
done

# The docs cite machine-readable artifacts (experiment logs, result CSVs, self-check summaries, figures), so a
# submission folder without them asserts things it cannot show. Copy the small evidence files; skip weights,
# feature banks and the training-run scratch dirs, which are reproducible rather than evidentiary.
cd DL_PBL || exit 1
find outputs/logs -type f \( -name "*.jsonl" -o -name "*.csv" -o -name "*.md" -o -name "*.txt" -o -name "*.png" \) \
  -not -path "outputs/logs/probe/*" -not -path "outputs/logs/yolo_runs/*" \
  -not -path "outputs/logs/yolo_runs_dedup/*" -not -path "outputs/logs/smoke2/*" \
  -not -path "outputs/logs/_app_test/*" -not -path "outputs/logs/notebook07_smoke/*" \
  -not -path "outputs/logs/notebook06_cpu/*" -print0 |
while IFS= read -r -d '' f; do
  mkdir -p "$DEST/$(dirname "$f")" && cp "$f" "$DEST/$f"
done
cd "$ROOT" || exit 1

# Audit the auditor: every artifact the claim checker reads must exist inside the frozen copy, otherwise the
# snapshot would be a set of documents whose own evidence is absent.
"$PY" - "$DEST" "$ROOT" <<'PY'
import re, sys
from pathlib import Path
dest, root = Path(sys.argv[1]), Path(sys.argv[2])
src = (root / "DL_PBL/scripts/verify_claims.py").read_text(encoding="utf-8")
cited = set(re.findall(r'"(DL_PBL/[^"]+)"', src))
missing = sorted(c for c in cited
                 if (root / c).is_file() and not (dest / c.removeprefix("DL_PBL/")).is_file())
print(f"artifacts cited by the checker: {len(cited)} | missing from the snapshot: {len(missing)}")
for m in missing[:12]:
    print("  MISSING", m)
sys.exit(1 if missing else 0)
PY
[ $? -eq 0 ] || { echo "snapshot discarded (cited artifacts missing)"; rm -rf "$DEST"; exit 1; }

# Notebooks are the deliverable; refuse to snapshot one that errored, never ran, or whose code cells
# cannot even compile.
# A refused snapshot must leave nothing behind: an incomplete folder with no MANIFEST.json looks exactly like
# a submission candidate in a directory listing, and only the manifest proves the gates passed.
discard() { echo "snapshot discarded ($1 refused it): $DEST"; rm -rf "$DEST"; exit 1; }
"$PY" "$ROOT/DL_PBL/scripts/check_notebooks.py" || discard check_notebooks
# Numbers quoted in the docs must still match the artifacts that produced them.
"$PY" "$ROOT/DL_PBL/scripts/verify_claims.py" || discard verify_claims
"$PY" - "$DEST" <<'PY'
import hashlib, json, sys
from pathlib import Path

dest = Path(sys.argv[1])
bad, files = [], {}
for nb_path in sorted((dest / "notebooks").rglob("*.ipynb")):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    errors = [f"{nb_path.name}:{i}" for i, c in enumerate(nb.get("cells", []))
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    code_cells = sum(1 for c in nb.get("cells", []) if c.get("cell_type") == "code")
    executed = sum(1 for c in nb.get("cells", []) if c.get("cell_type") == "code" and c.get("execution_count"))
    if errors:
        bad.append((nb_path.name, errors))
    if code_cells and executed != code_cells:
        bad.append((nb_path.name, f"only {executed}/{code_cells} code cells executed"))

print(f"notebooks checked: {len(list((dest / 'notebooks').rglob('*.ipynb')))}")
if bad:
    for name, why in bad:
        print("REFUSED:", name, why, file=sys.stderr)
    sys.exit(1)

for p in sorted(dest.rglob("*")):
    if p.is_file():
        files[str(p.relative_to(dest))] = {"bytes": p.stat().st_size,
                                          "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
manifest = {"snapshot_utc": str(__import__("datetime").datetime.now(__import__("datetime").timezone.utc)),
            "python": sys.version.split()[0], "files": files, "file_count": len(files)}
(dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"snapshot frozen: {len(files)} files -> {dest}")
PY
STATUS=$?
[ $STATUS -ne 0 ] && rm -rf "$DEST" && echo "snapshot discarded (gate refused it)"
exit $STATUS
