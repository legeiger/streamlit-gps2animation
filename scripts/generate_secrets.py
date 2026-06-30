from __future__ import annotations

import argparse
from pathlib import Path


TEMPLATE = """# Copy this file to `.streamlit/secrets.toml` or run `python scripts/generate_secrets.py`.

[garmin]
# Garmin Connect email address used for your personal account.
EMAIL = "{email}"

# Garmin Connect password. Keep this file private and out of version control.
PASSWORD = "{password}"
"""


def prompt(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw or default


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate .streamlit/secrets.toml for Track Forge")
    parser.add_argument("--output", default=".streamlit/secrets.toml", help="Path to write the secrets file")
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it already exists")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise SystemExit(f"{output_path} already exists. Use --force to overwrite it.")

    print("Track Forge secrets generator")
    print("Press Enter to accept the suggested default for each value.")

    email = prompt("Garmin EMAIL", "your-email@example.com")
    password = prompt("Garmin PASSWORD", "your-garmin-password")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(TEMPLATE.format(email=email, password=password), encoding="utf-8")

    print(f"Wrote {output_path}")
    print("If you use Git, keep `.streamlit/secrets.toml` out of version control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())