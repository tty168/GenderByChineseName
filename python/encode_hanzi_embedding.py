#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np


DEFAULT_EMBEDDINGS = "hanzi_embeddings_64d.npy"
DEFAULT_LOOKUP = "hanzi_index_lookup.txt"


def load_embeddings(
    embeddings_file=DEFAULT_EMBEDDINGS,
    lookup_file=DEFAULT_LOOKUP,
):
    """
    Load the precomputed Hanzi embeddings and the
    character-to-row lookup.

    hanzi_index_lookup.txt is a SINGLE comma-separated
    line written by chinese_char_autoencoder.py:

        f.write(",".join(char_index.tolist()))

    Therefore the position of a character in this list
    is its row index in hanzi_embeddings_64d.npy.
    """

    embeddings = np.load(embeddings_file)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2-D embedding matrix, "
            f"got {embeddings.shape}"
        )

    if embeddings.shape[1] != 64:
        raise ValueError(
            f"Expected 64-D embeddings, "
            f"got shape {embeddings.shape}"
        )

    with open(
        lookup_file,
        "r",
        encoding="utf-8"
    ) as f:
        content = f.read().strip()

    characters = [
        c.strip()
        for c in content.split(",")
        if c.strip()
    ]

    if len(characters) != len(embeddings):
        raise ValueError(
            "Embedding/lookup size mismatch: "
            f"{len(embeddings)} embedding rows but "
            f"{len(characters)} characters in lookup."
        )

    lookup = {
        character: index
        for index, character in enumerate(characters)
    }

    return embeddings, lookup


def encode_hanzi(
    character,
    embeddings,
    lookup,
):
    """
    Return the precomputed 64-D embedding for one
    Chinese character.
    """

    if not isinstance(character, str):
        raise TypeError(
            "character must be a string"
        )

    if len(character) != 1:
        raise ValueError(
            f"Expected exactly one character, "
            f"got {character!r}"
        )

    if character not in lookup:
        raise KeyError(
            f"Character {character!r} "
            "not found in hanzi_index_lookup.txt"
        )

    index = lookup[character]

    return embeddings[index]


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Look up precomputed 64-D Hanzi embeddings "
            "from hanzi_embeddings_64d.npy."
        )
    )

    parser.add_argument(
        "characters",
        nargs="?",
        help=(
            "Chinese character or text. "
            "If omitted, read from stdin."
        ),
    )

    parser.add_argument(
        "--embeddings",
        default=DEFAULT_EMBEDDINGS,
        help="Path to hanzi_embeddings_64d.npy",
    )

    parser.add_argument(
        "--lookup",
        default=DEFAULT_LOOKUP,
        help="Path to hanzi_index_lookup.txt",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output embeddings as JSON.",
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print only embedding values.",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load embedding matrix and lookup
    # -----------------------------------------------------

    try:
        embeddings, lookup = load_embeddings(
            args.embeddings,
            args.lookup,
        )
    except Exception as e:
        parser.error(
            f"Could not load embeddings: {e}"
        )

    # -----------------------------------------------------
    # Read input
    # -----------------------------------------------------

    if args.characters is not None:
        text = args.characters

    else:
        if sys.stdin.isatty():
            parser.error(
                "No character supplied. "
                "Provide a character as an argument "
                "or pipe text through stdin."
            )

        text = sys.stdin.read()

    # Remove whitespace but preserve character order.
    characters = [
        c for c in text
        if not c.isspace()
    ]

    if not characters:
        parser.error("No characters found in input.")

    # -----------------------------------------------------
    # Encode
    # -----------------------------------------------------

    results = []

    for character in characters:

        try:

            embedding = encode_hanzi(
                character,
                embeddings,
                lookup,
            )

            results.append({
                "character": character,
                "embedding": embedding.tolist(),
            })

        except KeyError as e:

            print(
                f"WARNING: {e}",
                file=sys.stderr,
            )

    if not results:
        sys.exit(1)

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    if args.json:

        print(
            json.dumps(
                results,
                ensure_ascii=False,
                indent=2,
            )
        )

    elif args.compact:

        for result in results:
            print(
                " ".join(
                    f"{x:.8f}"
                    for x in result["embedding"]
                )
            )

    else:

        for result in results:

            print(
                f"{result['character']}:"
            )

            print(
                " ".join(
                    f"{x:.8f}"
                    for x in result["embedding"]
                )
            )

            print()


if __name__ == "__main__":
    main()