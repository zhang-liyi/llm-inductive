"""One-shot script: call Gemini to generate 50 additional sports subdomains,
then append them to sports-subdomains.txt to reach 100 total."""

import os
import re

from google import genai
from google.genai import types

os.environ['GEMINI_API_KEY'] = '<GEMINI_API_KEY_PLACEHOLDER>'
client = genai.Client()

_DIR = os.path.dirname(__file__)
SUBDOMAINS_FILE = os.path.join(_DIR, "sports-subdomains.txt")


def main():
    with open(SUBDOMAINS_FILE) as f:
        existing_raw = f.read().strip()
    existing = [s.strip() for s in existing_raw.split(",") if s.strip()]
    print(f"Current count: {len(existing)}")

    prompt = (
        "Here is a list of sports subdomains already in use:\n"
        f"{', '.join(existing)}\n\n"
        "Generate exactly 50 more sports or athletic competition subdomains that are "
        "NOT already in the list above. These should be real sports or athletic disciplines, "
        "including niche, Paralympic, e-sport-adjacent physical activities, martial arts, "
        "water sports, precision sports, and traditional/folk sports from around the world. "
        "Return ONLY a comma-separated list of the 50 new sports, with no numbering, "
        "no explanations, and no extra text."
    )

    config = types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",
        temperature=0.7,
        max_output_tokens=1024,
    )
    resp = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt,
        config=config,
    )
    new_raw = resp.text.strip()

    # Parse response: split on commas, strip whitespace/numbering
    new_sports = [re.sub(r"^\d+[\.\)]\s*", "", s).strip()
                  for s in new_raw.split(",") if s.strip()]
    # Deduplicate against existing (case-insensitive)
    existing_lower = {s.lower() for s in existing}
    new_sports = [s for s in new_sports if s.lower() not in existing_lower]

    print(f"New sports from Gemini: {len(new_sports)}")
    print(", ".join(new_sports))

    combined = existing + new_sports
    print(f"Total after merge: {len(combined)}")

    with open(SUBDOMAINS_FILE, "w") as f:
        f.write(", ".join(combined) + "\n")
    print(f"Updated {SUBDOMAINS_FILE}")


if __name__ == "__main__":
    main()
