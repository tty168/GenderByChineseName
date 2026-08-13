import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MAX_SEQ_LEN = 12
LATENT_DIM = 64


# ---------------------------------------------------------
# Load vocabularies
# ---------------------------------------------------------

saved_rad_vocab = np.load(
    "saved_radical_vocab.npy",
    allow_pickle=True
).tolist()

saved_comp_vocab = np.load(
    "saved_component_vocab.npy",
    allow_pickle=True
).tolist()


# ---------------------------------------------------------
# Recreate the exact inference vocabularies
# ---------------------------------------------------------

radical_lookup = layers.StringLookup(
    vocabulary=saved_rad_vocab,
    output_mode="int",
    oov_token="UNK"
)


def split_by_individual_characters(input_string):
    return tf.strings.unicode_split(
        input_string,
        input_encoding="UTF-8"
    )


sequence_vectorizer = layers.TextVectorization(
    standardize=None,
    split=split_by_individual_characters,
    output_mode="int",
    output_sequence_length=MAX_SEQ_LEN,
    vocabulary=saved_comp_vocab
)


num_radicals = radical_lookup.vocabulary_size()
num_components = sequence_vectorizer.vocabulary_size()


# ---------------------------------------------------------
# Rebuild EXACT pretrained encoder architecture
# ---------------------------------------------------------

enc_rad_in = layers.Input(
    shape=(1, 1),
    dtype=tf.int32,
    name="enc_rad_idx"
)

enc_seq_in = layers.Input(
    shape=(1, MAX_SEQ_LEN),
    dtype=tf.int32,
    name="enc_seq_idx"
)

rad_squeezed = layers.Reshape((1,))(enc_rad_in)
seq_squeezed = layers.Reshape((MAX_SEQ_LEN,))(enc_seq_in)

rad_embed = layers.Embedding(
    input_dim=num_radicals,
    output_dim=32
)(rad_squeezed)

rad_features = layers.Flatten()(rad_embed)

seq_embed = layers.Embedding(
    input_dim=num_components,
    output_dim=64
)(seq_squeezed)

seq_conv = layers.Conv1D(
    filters=64,
    kernel_size=3,
    padding="same",
    activation="relu"
)(seq_embed)

seq_features = layers.GlobalAveragePooling1D()(seq_conv)

fused_features = layers.Concatenate()([
    rad_features,
    seq_features
])

latent_embedding = layers.Dense(
    LATENT_DIM,
    activation=None,
    name="hanzi_latent_space"
)(fused_features)

encoder = Model(
    inputs=[enc_rad_in, enc_seq_in],
    outputs=latent_embedding,
    name="Pretrained_Hanzi_Encoder"
)


# ---------------------------------------------------------
# Load pretrained weights
# ---------------------------------------------------------

encoder.load_weights(
    "hanzi_encoder_weights.weights.h5"
)

encoder.trainable = False


# ---------------------------------------------------------
# Load character dictionary
# ---------------------------------------------------------

def load_hanzi_dictionary(
    file_path="./hanzi_dictionary.csv"
):
    df = pd.read_csv(
        file_path,
        encoding="utf-8"
    )

    # Make character the lookup key.
    #
    # The exact column names should be checked against
    # the current CSV. This allows the function to tolerate
    # either "character" or "char".
    if "character" in df.columns:
        char_column = "character"
    elif "char" in df.columns:
        char_column = "char"
    else:
        raise ValueError(
            "Cannot find character column in hanzi_dictionary.csv"
        )

    df[char_column] = df[char_column].astype(str)

    return df.set_index(char_column)


hanzi_dictionary = load_hanzi_dictionary()


# ---------------------------------------------------------
# Character encoder
# ---------------------------------------------------------

def encode_hanzi(character):
    """
    Encode one Chinese character into its pretrained
    64-dimensional sub-character representation.

    Parameters
    ----------
    character : str
        A single Chinese character.

    Returns
    -------
    np.ndarray
        Shape: (64,)

    """

    if not isinstance(character, str):
        raise TypeError(
            "character must be a string"
        )

    if len(character) != 1:
        raise ValueError(
            f"Expected exactly one character, got: {character!r}"
        )

    if character not in hanzi_dictionary.index:
        raise KeyError(
            f"Character {character!r} "
            "was not found in hanzi_dictionary.csv"
        )

    row = hanzi_dictionary.loc[character]

    # -----------------------------------------------------
    # Obtain radical and ordered decomposition
    # -----------------------------------------------------

    # Adjust these two names if the CSV uses different
    # column names.
    radical = row["radical"]
    decomposition = row["decomposition"]

    radical = str(radical).strip()
    decomposition = str(decomposition).strip()

    # -----------------------------------------------------
    # Convert radical to vocabulary ID
    # -----------------------------------------------------

    radical_id = (
        radical_lookup(
            np.array([[radical]])
        )
        .numpy()
        .astype(np.int32)
        .reshape(1, 1)
    )

    # -----------------------------------------------------
    # Convert ordered decomposition into sequence IDs
    # -----------------------------------------------------

    sequence_ids = (
        sequence_vectorizer(
            np.array([[decomposition]])
        )
        .numpy()
        .astype(np.int32)
        .reshape(1, MAX_SEQ_LEN)
    )

    # -----------------------------------------------------
    # Run frozen encoder
    # -----------------------------------------------------

    embedding = encoder(
        [
            radical_id.reshape(1, 1, 1),
            sequence_ids.reshape(1, 1, MAX_SEQ_LEN)
        ],
        training=False
    )

    return embedding.numpy().reshape(LATENT_DIM)

# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Encode Chinese characters using the pretrained "
            "radical + ordered-structure Hanzi encoder."
        )
    )

    parser.add_argument(
        "characters",
        nargs="?",
        help=(
            "Chinese character(s) to encode. "
            "If omitted, characters are read from stdin."
        )
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output embeddings as JSON."
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Print one embedding per line without "
            "character labels."
        )
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Get input
    # -----------------------------------------------------

    if args.characters is not None:
        text = args.characters
    else:
        if sys.stdin.isatty():
            parser.error(
                "No character supplied. Provide a character "
                "as an argument or pipe text through stdin."
            )

        text = sys.stdin.read()

    # Remove whitespace/newlines but preserve character order
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
            embedding = encode_hanzi(character)

            results.append({
                "character": character,
                "embedding": embedding.tolist()
            })

        except KeyError as e:
            print(
                f"WARNING: {e}",
                file=sys.stderr
            )

        except Exception as e:
            print(
                f"ERROR encoding {character!r}: {e}",
                file=sys.stderr
            )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    if args.json:

        print(
            json.dumps(
                results,
                ensure_ascii=False,
                indent=2
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
