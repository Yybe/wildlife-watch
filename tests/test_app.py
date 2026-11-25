"""Input-validation checks for the alert prototype - no model, no GPU.

Run: .venv/Scripts/python.exe DL_PBL/tests/test_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/app"))
sys.path.insert(0, str(ROOT / "src/sequence"))
import predict as P  # noqa: E402

tmp = ROOT / "outputs/logs/_app_test"
tmp.mkdir(parents=True, exist_ok=True)
rejected = 0

for name, make in [("missing.jpg", None),
                   ("tiny.jpg", lambda p: p.write_bytes(b"x" * 10)),
                   ("wrongext.txt", lambda p: p.write_bytes(b"x" * 300)),
                   ("corrupt.jpg", lambda p: p.write_bytes(b"not an image " * 30)),
                   ("toolarge.jpg", lambda p: p.write_bytes(b"0" * (25 * 2**20 + 1)))]:
    path = tmp / name
    if make:
        make(path)
    elif path.exists():
        path.unlink()
    try:
        P.validate_image(str(path))
    except ValueError:
        rejected += 1
    else:
        raise AssertionError(f"accepted bad input: {name}")

good = tmp / "real.jpg"
from PIL import Image  # noqa: E402
Image.new("RGB", (64, 48), (10, 20, 30)).save(good, "JPEG")
assert P.validate_image(str(good)) == good, "rejected a valid image"

# A person-shaped file name must not be trusted: only magic-byte decoding decides.
fake = tmp / "photo.jpg.exe"
fake.write_bytes(b"MM" + b"0" * 300)
try:
    P.validate_image(str(fake))
    raise AssertionError("extension spoof accepted")
except ValueError:
    rejected += 1

print(f"test_app: {rejected + 1} checks passed ({rejected} bad inputs rejected, 1 valid accepted)")
