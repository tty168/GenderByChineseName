# Efficient Sub-Character Transfer Learning for Chinese Name Gender Prediction

**A compact Chinese character encoder based on radical identity and
ordered structural decomposition, pretrained with an autoencoder and
transferred to downstream name-level prediction.**

------------------------------------------------------------------------

## Overview

This repository explores a practical approach to **Chinese sub-character representation learning and transfer learning**.

Instead of treating each Chinese character as an atomic symbol, the model represents a character using two complementary structural signals:

1.  **Radical identity**
2.  **Ordered structural decomposition**, including Ideographic
    Description Characters (IDCs) and their component sequence

These inputs are encoded by a lightweight autoencoder into a
**64-dimensional character representation**.

The pretrained encoder is then **frozen** and reused for a downstream
Chinese name gender prediction task.

The central idea is:

    Chinese character
           │
           ├── Radical
           │
           └── Ordered structural decomposition
                        │
                        ▼
                 Structural Encoder
                        │
                        ▼
                     64-D vector
                        │
                 ┌──────┴──────┐
                 │             │
            reconstruction   transfer
                               │
                               ▼
                     Chinese name model

For a two-character name:

$$ z_1 = E(c_1) ^{64} $$

$$ z_2 = E(c_2) ^{64} $$

and the name representation is:

$$ z_{} = [z_1;z_2] ^{128}. $$

The downstream predictor therefore learns from **reusable sub-character structural representations**, rather than learning a separate embedding for every complete name.

------------------------------------------------------------------------

## Research Question

The central question is:

> **How much useful Chinese character information can be compressed into
> a small, reusable representation when the model is given both radical
> identity and ordered structural decomposition?**

The corresponding hypothesis is:

> **A Chinese character representation learned from sub-character
> structure can transfer to a downstream prediction task without
> requiring the pretrained encoder to be fine-tuned.**

The current repository uses Chinese name gender prediction as a
downstream demonstration of this transfer.

------------------------------------------------------------------------

# Why Ordered Structural Decomposition?

Chinese character representations based only on component membership
lose some structural information.

For example:

    ⿰ A B

contains:

- the structural operator `⿰`;
- component `A`;
- component `B`;
- the ordering of those components.

An unordered representation such as:

    {A, B}

does not preserve the same information.

The model in this repository therefore treats the decomposition as an
**ordered sequence**, rather than simply as an unordered collection of
components.

Conceptually:

    Radical
       +
    Ordered IDS / structural sequence
       ↓
    Compact structural representation

This is an important distinction from a simple bag-of-components
representation.

### Novelty positioning

This project does **not** claim that radical embeddings, component
embeddings, or IDS representations are themselves new. These are
established forms of Chinese sub-character information.

The contribution explored here is the practical combination of:

- radical identity;
- ordered structural decomposition;
- compact 64-dimensional bottleneck;
- multi-target structural reconstruction;
- frozen encoder transfer;
- position-preserving composition of character embeddings for names.

The intended contribution is therefore **architectural and empirical**,
rather than the invention of the underlying Chinese character
decomposition concepts.

------------------------------------------------------------------------

# Architecture

## Stage 1 --- Structural Autoencoder

The character encoder receives:

    radical + ordered structural decomposition sequence

The two representations are processed separately and then fused.

                      Character
                         │
              ┌──────────┴──────────┐
              │                     │
           Radical             Structural Sequence
              │                     │
              ▼                     ▼
       Radical Embedding       Token Embedding
              │                     │
              │                   Conv1D
              │                     │
              │              Global Average Pooling
              │                     │
              └──────────┬──────────┘
                         │
                    Concatenate
                         │
                         ▼
                     Dense(64)
                         │
                         ▼
                    z ∈ R⁶⁴
                         │
                  ┌──────┴──────┐
                  │             │
                  ▼             ▼
           Radical Decoder   Sequence Decoder
                  │             │
                  ▼             ▼
            Radical output   Structural output

The encoder is trained through reconstruction rather than directly
through the downstream gender labels.

This makes the learned 64-dimensional vector a **structural
representation** rather than a gender-specific embedding.

------------------------------------------------------------------------

## Stage 2 --- Frozen Transfer Learning

After structural pretraining, the encoder is frozen.

For a two-character name:

    Character 1
        │
        ▼
     Frozen Encoder
        │
        ▼
      64-D
        │
        ├──────────────┐
                       │
    Character 2        │
        │              │
        ▼              │
     Frozen Encoder    │
        │              │
        ▼              │
      64-D             │
        │              │
        └──────┬───────┘
               ▼
          Concatenate
               │
               ▼
            128-D
               │
               ▼
        Gender Predictor

Mathematically:

$$ z(c)=E(r_c,s_c) $$

where:

- (r_c) is the radical representation;
- (s_c) is the ordered structural sequence;
- (E) is the pretrained encoder.

For a two-character name:

$$ z_{\mathrm{name}}=[z(c_1),z(c_2)] $$

The downstream model then learns:

$$ P(name) = G(z_{\mathrm{name}}). $$

The encoder remains frozen during downstream training.

------------------------------------------------------------------------

# Repository Structure

The important files are:

    .
    ├── data/
    │   ├── radicals.txt
    │   ├── components.txt
    │   ├── training_dataset.txt
    │   └── hanzi.db
    │
    ├── python/
    │   ├── scan_schema.py
    │   ├── chinese_char_autoencoder.py
    │   └── chinese_name_gender_predictor.py
    │
    ├── hanzi_encoder_weights.weights.h5
    ├── saved_radical_vocab.npy
    ├── saved_component_vocab.npy
    ├── hanzi_embeddings_64d.npy
    ├── hanzi_index_lookup.txt
    └── hanzi_name_gender_predictor.keras

------------------------------------------------------------------------

# Character Data

The repository contains a structural inventory of approximately **9,574
Chinese characters**.

## Radical vocabulary

    data/radicals.txt

contains approximately **294 radical entries**.

## Component vocabulary

    data/components.txt

contains approximately **1,823 component entries**.

## Character training data

    data/training_dataset.txt

contains character records with:

    character | radical | decomposition

The decomposition field preserves the structural sequence used by the
encoder.

The decomposition may contain structural operators and component
characters. These values are used as input features for the structural
encoder and its reconstruction objective.

------------------------------------------------------------------------

# SQLite Character Database

    data/hanzi.db

contains detailed information for approximately 9,574 characters.

The database includes character-level structural information and the
Unicode Ideographic Description Characters.

For example:

    ⿰  U+2FF0  left to right
    ⿱  U+2FF1  above to below
    ⿲  U+2FF2  left to middle and right
    ⿳  U+2FF3  above to middle and below
    ⿴  U+2FF4  full surround

The database can be inspected with SQLite:

    sqlite3 data/hanzi.db ".tables"
    sqlite3 data/hanzi.db ".schema characters"
    sqlite3 data/hanzi.db \
      "select * from characters limit 10"

------------------------------------------------------------------------

# Data Processing

The character database is generated from the source dictionary data.

    python/scan_schema.py

reads the source character information and generates the structured CSV
used to create the SQLite database.

The resulting character schema includes fields such as:

    character
    definition
    pinyin
    decomposition
    radical
    matches
    etymology_hint
    etymology_phonetic
    etymology_semantic
    etymology_type

------------------------------------------------------------------------

# Installation

The current implementation uses Python 3.12 with TensorFlow/Keras.

## Create environment

    conda create -n keras3_env python=3.12 -y
    conda activate keras3_env

## Install dependencies

    pip install --upgrade pip
    pip install tensorflow==2.21.0 keras==3.15.1 numpy

Verify Keras:

    python -c \
    "import os; os.environ['KERAS_BACKEND']='tensorflow'; \
    import keras; print(keras.__version__)"

------------------------------------------------------------------------

# Step 1 --- Train the Character Autoencoder

Run:

    python python/chinese_char_autoencoder.py

### Inputs

The autoencoder uses:

    data/radicals.txt
    data/components.txt
    data/training_dataset.txt

### Main outputs

The encoder weights and vocabularies are saved as:

    hanzi_encoder_weights.weights.h5
    saved_radical_vocab.npy
    saved_component_vocab.npy

The script also generates a precomputed embedding cache:

    hanzi_embeddings_64d.npy
    hanzi_index_lookup.txt

------------------------------------------------------------------------

# Encoder Model vs. `hanzi_embeddings_64d.npy`

There are two related but different artifacts.

## 1. Encoder + vocabularies

    hanzi_encoder_weights.weights.h5
    saved_radical_vocab.npy
    saved_component_vocab.npy

These constitute the **general reusable encoder**.

They can be used to compute:

$$ (r_c,s_c)\rightarrow z_c $$

for characters represented by the structural input vocabulary.

This is the preferred representation of the trained model for transfer
learning.

## 2. Precomputed embedding matrix

    hanzi_embeddings_64d.npy

is a cached matrix containing the 64-dimensional embeddings already
generated for the characters in the training inventory.

It is useful for fast lookup:

    character
        ↓
    embedding lookup
        ↓
    64-D vector

The `.npy` file is therefore a **cache of model outputs**, not a
replacement for the encoder itself.

If the goal is to build a more general downstream system, retain the
encoder weights and vocabularies.

------------------------------------------------------------------------

# Step 2 --- Train the Name Gender Predictor

Run:

    python python/chinese_name_gender_predictor.py

The downstream model uses the frozen pretrained character encoder.

## Inputs

    hanzi_encoder_weights.weights.h5
    saved_radical_vocab.npy
    saved_component_vocab.npy
    data/training_dataset.txt

and the external CnGender name dataset.

The resulting model is:

    hanzi_name_gender_predictor.keras

------------------------------------------------------------------------

# Why Does the Gender Predictor Need `training_dataset.txt`?

The downstream predictor does not need `training_dataset.txt` to retrain
the character encoder.

The encoder is already pretrained and frozen.

Instead, `training_dataset.txt` serves as the **character → structural
decomposition lookup** needed to convert the characters appearing in a
name into the inputs expected by the encoder.

Conceptually:

    Chinese name
         │
         ▼
    individual characters
         │
         ▼
    training_dataset.txt
         │
         ├── radical
         │
         └── ordered decomposition
                 │
                 ▼
           Frozen Encoder
                 │
                 ▼
               64-D

Therefore:

> `training_dataset.txt` is part of the **input representation
> vocabulary/lookup pipeline**, not part of the downstream
> label-learning process.

This distinction is important when deploying the encoder independently.

------------------------------------------------------------------------

# CnGender Dataset

The downstream experiment uses the **Chinese Name-to-Gender Dataset
(CnGender)**.

The repository references the dataset as an external dependency rather
than redefining or redistributing the source dataset.

The dataset contains approximately **one million Chinese names** with
gender probability information.

The predictor uses the name and its associated male-probability target
for downstream learning.

See the dataset source and paper references at the end of this README.

------------------------------------------------------------------------

# Preliminary Results

The current implementation reports the following preliminary evaluation:

| Metric | Result |
| :--- | ---: |
| Accuracy | **92.17%** |
| ROC-AUC | **0.9683** |
| Evaluation records | **20,000** |

These numbers should be interpreted as **preliminary measurements**, not
as definitive benchmark results.

The current experiment uses a 20,000-record evaluation subset of the
CnGender data. A future version should establish a strictly
non-overlapping train/validation/test split before treating the test
results as final generalization estimates.

The results provide an initial indication that the frozen structural
representation retains useful discriminative information for the
downstream name-level task.

They do **not**, by themselves, establish which particular structural
features are responsible for the performance.

------------------------------------------------------------------------

# What This Experiment Is Testing

The downstream experiment tests whether:

    Chinese character structure
            ↓
    self-supervised encoder
            ↓
    frozen 64-D representation
            ↓
    name-level prediction

can transfer useful information without fine-tuning the character
encoder.

The key research question is therefore broader than gender prediction:

> **Can a compact structural representation learned independently of a
> downstream task provide reusable information for later tasks?**

Gender prediction is the current demonstration task.

------------------------------------------------------------------------

# Most Important Future Experiment

The most direct test of the proposed representation is an
**ordered-versus-unordered structural ablation**.

Compare:

    Radical + unordered components

against:

    Radical + ordered structural sequence

The hypothesis is:

$$ \text{Radical + ordered structure} > \text{Radical + unordered components} $$

under comparable model capacity.

Additional useful ablations include:

    1. Random character embedding
    2. Radical only
    3. Ordered structure only
    4. Unordered components only
    5. Radical + unordered components
    6. Radical + ordered structure
    7. Frozen encoder
    8. Fine-tuned encoder

A bottleneck experiment can also compare:

    16-D
    32-D
    64-D
    128-D
    256-D

to determine how much structural information can be retained in a
compact representation.

------------------------------------------------------------------------

# Efficiency

The character representation is deliberately compact:

$$ z_c\in\mathbb{R}^{64}. $$

A two-character name requires:

$$ z_{\mathrm{name}}\in\mathbb{R}^{128}. $$

This makes the representation attractive for applications that require
repeated processing of large numbers of Chinese names or characters.

The architecture also separates:

    structural representation learning

from:

    downstream task learning.

Once the encoder has been trained, the same character representation can
potentially be reused for multiple downstream tasks.

------------------------------------------------------------------------

# Limitations

The current implementation should be viewed as a research prototype.

Important limitations include:

### 1. Sequence rather than explicit tree encoding

The structural decomposition is processed as an ordered sequence.

The encoder does not explicitly construct a recursive IDS tree.

### 2. Orthographic representation

The current representation emphasizes written character structure. It
does not explicitly model Pinyin, tone, or other phonological
information.

### 3. Unknown characters

Characters absent from the structural vocabulary require fallback
handling.

### 4. Preliminary downstream evaluation

The current 20,000-record evaluation should not be interpreted as a
definitive held-out benchmark until a strict train/validation/test
protocol is established.

### 5. No ablation yet

The current result does not isolate the contribution of:

- radical information;
- component identity;
- structural ordering;
- structural operators;
- the 64-dimensional bottleneck.

These should be tested in future experiments.

------------------------------------------------------------------------

# Reproducibility

The core experiment can be reproduced from the repository using:

    Character structural data
            ↓
    scan_schema.py
            ↓
    hanzi.db / training_dataset.txt
            ↓
    chinese_char_autoencoder.py
            ↓
    frozen encoder
            ↓
    chinese_name_gender_predictor.py
            ↓
    name-level prediction

The principal model artifacts are:

    hanzi_encoder_weights.weights.h5
    saved_radical_vocab.npy
    saved_component_vocab.npy
    hanzi_embeddings_64d.npy
    hanzi_index_lookup.txt
    hanzi_name_gender_predictor.keras

------------------------------------------------------------------------

# External Data and Dependencies

This repository uses external character and name datasets.

## Chinese character data

The character decomposition data is based on publicly available Chinese
character resources.

The repository also includes processing derived from Make Me a Hanzi:

- `dictionary.txt`

## Chinese name gender data

The downstream experiment uses:

**CnGender --- Chinese Name-to-Gender Dataset**

The dataset contains large-scale Chinese name observations and gender
probability information.

Please consult the original dataset source for its licensing and usage
requirements.

------------------------------------------------------------------------

# References

1.  **Chinese Character Components**\
    https://en.wikipedia.org/wiki/Chinese_character_components

2.  **Make Me a Hanzi**\
    https://github.com/skishore/makemeahanzi

3.  **Chinese Name-to-Gender Dataset (CnGender)**\
    https://www.nature.com/articles/s41597-025-05803-z

4.  **OSF Preprint**\
    https://doi.org/10.17605/OSF.IO/H2UJA

5.  **Radical-Enhanced Chinese Character Embedding**\
    https://arxiv.org/abs/1404.4714

6.  **Component-Enhanced Chinese Character Embeddings**\
    https://arxiv.org/abs/1508.06669

7.  **Sub-Character Tokenization for Chinese Pretrained Language
    Models**\
    https://aclanthology.org/2023.tacl-1.28/

------------------------------------------------------------------------

# Citation

If you use this repository or the structural encoder in your research,
please cite the associated OSF preprint:

    @misc{yuan2026subcharacter,
      author       = {Yuan, Ted},
      title        = {Efficient Sub-Character Transfer Learning for Chinese Name Gender Prediction},
      year         = {2026},
      publisher    = {OSF},
      doi          = {10.17605/OSF.IO/H2UJA},
      url          = {https://doi.org/10.17605/OSF.IO/H2UJA}
    }

------------------------------------------------------------------------

# Status

This repository represents an **active research prototype**.

The current results demonstrate the feasibility of transferring a
compact Chinese sub-character representation to a downstream name-level
prediction task.

The next major validation step is a controlled ablation study testing
whether **ordered structural decomposition provides measurable benefit
over unordered component representations**, followed by evaluation on a
strictly held-out test set.

The broader objective is to determine whether compact, structurally
informed Chinese character representations can provide a practical
alternative or complement to much larger language-model-based
representations for selected downstream tasks.
