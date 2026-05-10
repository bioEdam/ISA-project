import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = [
    # Application
    "app/__init__.py",
    "app/main.py",
    "app/models.py",
    "app/dependencies.py",
    "app/routers/__init__.py",
    "app/routers/recommender.py",
    "app/routers/playlists.py",
    "app/templates/index.html",
    "app/static/style.css",
    "app/static/api.js",
    "app/static/ui.js",
    "app/static/app.js",
    # Model & inference
    "src/models.py",
    "demo/recommender.py",
    # Database
    "db/schema.sql",
    "db/seed.py",
    "db/seed_data.parquet",
    # Scripts
    "scripts/entrypoint.sh",
    "scripts/retrain.py",
    "scripts/crontab",
    # Docker
    "Dockerfile",
    "docker-compose.yml",
    ".dockerignore",
    ".env.example",
    "requirements-app.txt",
    # Documentation
    "docs/installation.md",
    "docs/user_manual.md",
    "README.md",
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