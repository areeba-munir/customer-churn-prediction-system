"""Download and verify the original IBM Telco sample CSV."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

from src.data import DATA_PATH

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)


def verify_data(content: bytes) -> str:
    """Reject data that differs from the version used in this project."""
    fingerprint = hashlib.sha256(content).hexdigest()
    if fingerprint != EXPECTED_SHA256:
        raise ValueError(
            f"Dataset fingerprint mismatch: {fingerprint}. "
            "No existing file has been overwritten."
        )
    return fingerprint


def download_data(destination: Path) -> None:
    """Verify an existing file or download and verify a new one."""
    if destination.exists():
        fingerprint = verify_data(destination.read_bytes())
        print("Existing dataset verified; no changes made.")
    else:
        with urlopen(DATA_URL, timeout=60) as response:
            content = response.read()

        fingerprint = verify_data(content)
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Exclusive creation prevents overwriting an existing file.
        with destination.open("xb") as output:
            output.write(content)

        print("Dataset downloaded and verified.")

    print(f"Location: {destination}")
    print(f"SHA-256: {fingerprint}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_PATH,
        help="Optional download destination.",
    )
    args = parser.parse_args()
    download_data(args.output)


if __name__ == "__main__":
    main()