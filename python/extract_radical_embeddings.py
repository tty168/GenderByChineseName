#!/usr/bin/env python3

"""
Extract 64-D embeddings for Chinese radical characters.

Source:
    hanzi_embeddings_64d.npy
    hanzi_index_lookup.txt
    radicals.txt

The order of hanzi_index_lookup.txt is authoritative for mapping
embedding rows to characters.

The order of ./radicals.txt does NOT need to match the embedding
lookup order.

Output:
    radical_embeddings.csv

Each row contains:

    radical
    embedding

where embedding is a JSON-style list of 64 floating-point values.

Also produces:

    radical_embeddings_expanded.csv

with one column per embedding dimension:

    radical
    dim_01
    dim_02
    ...
    dim_64

The expanded version is convenient for pandas / R / ML analysis.
"""

from pathlib import Path
import argparse
import csv
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_EMBEDDINGS = "hanzi_embeddings_64d.npy"
DEFAULT_LOOKUP = "hanzi_index_lookup.txt"
DEFAULT_RADICALS = "./radicals.txt"

DEFAULT_OUTPUT = "radical_embeddings.csv"
DEFAULT_EXPANDED_OUTPUT = "radical_embeddings_expanded.csv"


# ---------------------------------------------------------------------
# Load embedding matrix
# ---------------------------------------------------------------------

def load_embeddings(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found:\n  {path}"
        )

    embeddings = np.load(path)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2-D embedding matrix, got: "
            f"{embeddings.shape}"
        )

    print(
        f"Embedding matrix: {embeddings.shape}"
    )

    return embeddings


# ---------------------------------------------------------------------
# Load character lookup
# ---------------------------------------------------------------------

def load_character_lookup(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Character lookup file not found:\n  {path}"
        )

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    # Current hanzi_index_lookup.txt is one
    # comma-separated line.
    chars = [
        ch.strip()
        for ch in text.split(",")
        if ch.strip()
    ]

    print(
        f"Character lookup: {len(chars):,} characters"
    )

    return chars


# ---------------------------------------------------------------------
# Load radicals
# ---------------------------------------------------------------------

def load_radicals(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Radical file not found:\n  {path}"
        )

    text = path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not text:
        return []

    # radicals.txt is comma-separated, possibly on one line.
    # Also tolerate newline-separated files.
    text = text.replace("\n", ",")
    text = text.replace("\r", ",")

    radicals = [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]

    # Remove duplicates while preserving original order.
    radicals = list(
        dict.fromkeys(radicals)
    )

    print(
        f"Radicals loaded: {len(radicals):,}"
    )

    return radicals

# ---------------------------------------------------------------------
# Extract embeddings
# ---------------------------------------------------------------------

def extract_radical_embeddings(
    embeddings,
    chars,
    radicals,
):

    if embeddings.shape[0] != len(chars):
        raise ValueError(
            "\nEmbedding / lookup mismatch:\n"
            f"  embeddings: {embeddings.shape[0]:,}\n"
            f"  lookup:     {len(chars):,}"
        )

    char_to_index = {
        char: i
        for i, char in enumerate(chars)
    }

    found = []
    missing = []

    for radical in radicals:

        if radical in char_to_index:

            index = char_to_index[radical]

            found.append(
                (
                    radical,
                    index,
                    embeddings[index],
                )
            )

        else:
            missing.append(radical)

    return found, missing


# ---------------------------------------------------------------------
# Save compact CSV
# ---------------------------------------------------------------------

def save_compact_csv(
    found,
    output_path,
):

    rows = []

    for radical, index, embedding in found:

        rows.append(
            {
                "radical": radical,
                "embedding": json.dumps(
                    embedding.astype(float).tolist(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return df


# ---------------------------------------------------------------------
# Save expanded CSV
# ---------------------------------------------------------------------

def save_expanded_csv(
    found,
    output_path,
):

    rows = []

    for radical, index, embedding in found:

        row = {
            "radical": radical
        }

        for i, value in enumerate(embedding):

            row[
                f"dim_{i + 1:02d}"
            ] = float(value)

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract radical-character embeddings "
            "from the Hanzi 64-D embedding matrix."
        )
    )

    parser.add_argument(
        "--embeddings",
        default=DEFAULT_EMBEDDINGS,
    )

    parser.add_argument(
        "--lookup",
        default=DEFAULT_LOOKUP,
    )

    parser.add_argument(
        "--radicals",
        default=DEFAULT_RADICALS,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--expanded-output",
        default=DEFAULT_EXPANDED_OUTPUT,
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print(
        "RADICAL CHARACTER EMBEDDING EXTRACTION"
    )
    print("=" * 80)

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    embeddings = load_embeddings(
        args.embeddings
    )

    chars = load_character_lookup(
        args.lookup
    )

    radicals = load_radicals(
        args.radicals
    )

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    print()
    print("Validating embedding / lookup alignment...")

    if embeddings.shape[0] != len(chars):

        raise ValueError(
            f"FAILED:\n"
            f"  embeddings = {embeddings.shape[0]:,}\n"
            f"  characters = {len(chars):,}"
        )

    print("Alignment: PASSED")

    # -------------------------------------------------------------
    # Extract
    # -------------------------------------------------------------

    print()
    print("Extracting radical embeddings...")

    found, missing = extract_radical_embeddings(
        embeddings,
        chars,
        radicals,
    )

    print(
        f"  Found   : {len(found):,}"
    )

    print(
        f"  Missing : {len(missing):,}"
    )

    # -------------------------------------------------------------
    # Save compact
    # -------------------------------------------------------------

    compact_df = save_compact_csv(
        found,
        args.output,
    )

    print()
    print(
        f"Saved compact CSV:"
    )

    print(
        f"  {args.output}"
    )

    print(
        f"  Rows: {len(compact_df):,}"
    )

    # -------------------------------------------------------------
    # Save expanded
    # -------------------------------------------------------------

    expanded_df = save_expanded_csv(
        found,
        args.expanded_output,
    )

    print()
    print(
        f"Saved expanded CSV:"
    )

    print(
        f"  {args.expanded_output}"
    )

    print(
        f"  Rows: {len(expanded_df):,}"
    )

    print(
        f"  Columns: {len(expanded_df.columns):,}"
    )

    # -------------------------------------------------------------
    # Missing radicals
    # -------------------------------------------------------------

    if missing:

        print()
        print("=" * 80)
        print("RADICALS NOT FOUND IN HANZI EMBEDDING VOCABULARY")
        print("=" * 80)

        for radical in missing:

            print(
                f"  {radical}"
            )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Embedding dimensions : "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Radicals requested    : "
        f"{len(radicals):,}"
    )

    print(
        f"Radicals found        : "
        f"{len(found):,}"
    )

    print(
        f"Radicals missing      : "
        f"{len(missing):,}"
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()
