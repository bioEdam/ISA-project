"""
Build a clean zip for the mini-project 3 submission.

Contains only the files needed to build and run the Docker image.
Run from the project root:
    python scripts/build_release_zip.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = [
    "app/main.py",
    "app/templates/index.html",
    "app/static/style.css",
    "app/static/app.js",
    "demo/recommender.py",
    "src/models.py",
    "Dockerfile",
    ".dockerignore",
    "requirements-app.txt",
    "docs/installation.md",
    "docs/user_manual.md",
]

OUT = ROOT / "model-deployment-code.zip"


def main():
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in FILES:
            path = ROOT / rel
            if not path.exists():
                print(f"  SKIP (missing): {rel}")
                continue
            zf.write(path, rel)
            print(f"  added: {rel}")

    print(f"\nCreated {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
