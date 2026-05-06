"""Extract the BACKGROUND section from a scenario file."""

import argparse
import sys


def extract_background(text: str) -> str:
    start = text.find("BACKGROUND")
    if start == -1:
        raise ValueError("No BACKGROUND section found")
    end = text.find("CONDITIONS", start)
    if end == -1:
        raise ValueError("No CONDITIONS section found")
    return text[start:end].rstrip()


def main():
    parser = argparse.ArgumentParser(description="Extract BACKGROUND section from a scenario file")
    parser.add_argument("file", help="Path to the scenario .txt file")
    args = parser.parse_args()

    with open(args.file) as f:
        text = f.read()

    print(extract_background(text))


if __name__ == "__main__":
    main()
