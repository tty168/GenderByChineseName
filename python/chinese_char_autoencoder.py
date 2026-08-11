import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import random

# ----------------------------------------------------
# 1. LOAD SINGLE-LINE VOCABULARIES
# ----------------------------------------------------
def load_comma_separated_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()
    return [token.strip() for token in raw_content.split(",") if token.strip()]

raw_rads = load_comma_separated_txt("radicals.txt")
raw_comps = load_comma_separated_txt("components.txt")

# Enforce clean unique boundaries for our vocab tracking indices
radical_vocab = ["PAD", "UNK"] + [r for r in raw_rads if r not in ["PAD", "UNK"]]
component_vocab = ["PAD", "UNK"] + [c for c in raw_comps if c not in ["PAD", "UNK"]]

MAX_SEQ_LEN = 12
LATENT_DIM = 64

# FIXED: Character-level custom splitting engine
# This function automatically converts "⿰丨？" into ['⿰', '丨', '？'] 
# without needing literal space characters inside your text file.
def split_by_individual_characters(input_string):
    return tf.strings.unicode_split(input_string, input_encoding="UTF-8")

# ----------------------------------------------------
# 2. STRING PREPROCESSING LAYER CONFIGURATION
# ----------------------------------------------------
radical_lookup = layers.StringLookup(vocabulary=radical_vocab, output_mode="int", oov_token="UNK")

# FIXED: Replaced split="whitespace" with our character-level splitting function
sequence_vectorizer = layers.TextVectorization(
    standardize=None, 
    split=split_by_individual_characters, 
    output_mode="int",
    output_sequence_length=MAX_SEQ_LEN, 
    vocabulary=component_vocab
)

num_radicals = radical_lookup.vocabulary_size()
num_components = sequence_vectorizer.vocabulary_size()
print(f"Embedding Space Boundaries -> Radicals: {num_radicals}, Components: {num_components}")

# ----------------------------------------------------
# 3. ENCODER BLOCK (Takes Integer Vectors Directly)
# ----------------------------------------------------
enc_rad_in = layers.Input(shape=(1,), dtype=tf.int32, name="input_radical")
enc_seq_in = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_sequence")

rad_embed = layers.Embedding(input_dim=num_radicals, output_dim=32)(enc_rad_in)
rad_features = layers.Flatten()(rad_embed)

seq_embed = layers.Embedding(input_dim=num_components, output_dim=64)(enc_seq_in)
seq_conv = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(seq_embed)
seq_features = layers.GlobalAveragePooling1D()(seq_conv)

fused_features = layers.Concatenate()([rad_features, seq_features])
latent_embedding = layers.Dense(LATENT_DIM, activation=None, name="hanzi_latent_space")(fused_features)
encoder_model = Model(inputs=[enc_rad_in, enc_seq_in], outputs=latent_embedding, name="Hanzi_Encoder")

# ----------------------------------------------------
# 4. DECODER BLOCK (Explicit Reshape Alignment)
# ----------------------------------------------------
decoder_in = layers.Input(shape=(LATENT_DIM,), name="latent_input")

dec_rad_dense = layers.Dense(64, activation="relu")(decoder_in)
rad_output = layers.Dense(num_radicals, activation="softmax", name="output_radical")(dec_rad_dense)

dec_seq_dense = layers.Dense(MAX_SEQ_LEN * 64, activation="relu")(decoder_in)
dec_seq_reshape = layers.Reshape((MAX_SEQ_LEN, 64))(dec_seq_dense)

# Transposed convolution path handles spatial sequence generation
dec_seq_conv = layers.Conv1DTranspose(filters=128, kernel_size=3, padding="same", activation="relu")(dec_seq_reshape)
sequence_output = layers.Dense(num_components, activation="softmax", name="output_sequence")(dec_seq_conv)

decoder_model = Model(
    inputs={"latent_input": decoder_in}, 
    outputs={"output_radical": rad_output, "output_sequence": sequence_output}, 
    name="Hanzi_Decoder"
)

# ----------------------------------------------------
# 5. FULL AUTOENCODER PIPELINE WITH DICTIONARY LOSS
# ----------------------------------------------------
autoencoder_outputs = decoder_model({"latent_input": encoder_model.output})
autoencoder_model = Model(inputs=encoder_model.input, outputs=autoencoder_outputs, name="Hanzi_Autoencoder")

autoencoder_model.compile(
    optimizer="adam",
    loss={
        "output_radical": "sparse_categorical_crossentropy", 
        "output_sequence": "sparse_categorical_crossentropy"
    },
    loss_weights={
        "output_radical": 1.0, 
        "output_sequence": 2.0  # Increased sequence loss weight to support sequence coherence
    }
)

# ----------------------------------------------------
# 6. DATA LOADING & VECTORIZATION
# ----------------------------------------------------
def load_sqlite_export(file_path, separator="|"):
    characters, radicals, sequences = [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(separator)
            if len(parts) == 3:
                char, rad, decomp = parts
                characters.append(char.strip())
                radicals.append(rad.strip())
                sequences.append(decomp.strip())
    return np.array(characters), np.array(radicals), np.array(sequences)

char_index, raw_input_rads, raw_input_seqs = load_sqlite_export("training_dataset.txt", separator="|")

raw_input_rads = raw_input_rads.reshape(-1, 1)
raw_input_seqs = raw_input_seqs.reshape(-1, 1)

# Pre-vectorize the text fields to extract clean input vectors
target_rads = radical_lookup(raw_input_rads).numpy().astype(np.int32)
target_seqs = sequence_vectorizer(raw_input_seqs).numpy().astype(np.int32)

# --- NEW DIAGNOSTIC PANEL VERIFICATION ---
print("\n--- Pre-vectorization Diagnostic Check ---")
for i in range(min(5, len(char_index))):
    print(f"Char: {char_index[i]} | Raw Seq: '{raw_input_seqs[i][0]}' -> Parsed IDs: {target_seqs[i]}")
print("------------------------------------------\n")

dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_radical": target_rads, 
        "input_sequence": target_seqs
    },
    {
        "output_radical": target_rads, 
        "output_sequence": target_seqs
    }
))
dataset = dataset.shuffle(buffer_size=2000).batch(64).prefetch(tf.data.AUTOTUNE)

# Train the model
autoencoder_model.fit(dataset, epochs=100, shuffle=None)

# ----------------------------------------------------
# 7. SAMPLING RECONSTRUCTION MODULE
# ----------------------------------------------------
def sample_reconstructions(encoder, decoder, raw_rads, raw_seqs, num_rads, num_seqs, char_labels, r_vocab, c_vocab, num_samples=5):
    print("\n" + "="*60)
    print(f"      SAMPLING {num_samples} RANDOM AUTOENCODER RECONSTRUCTIONS")
    print("="*60)
    
    total_records = len(char_labels)
    sample_indices = random.sample(range(total_records), min(num_samples, total_records))
    
    for idx in sample_indices:
        target_char = char_labels[idx]
        orig_rad = str(np.squeeze(raw_rads[idx]))
        orig_seq = str(np.squeeze(raw_seqs[idx]))
        
        test_rad_num = np.expand_dims(num_rads[idx], axis=0)
        test_seq_num = np.expand_dims(num_seqs[idx], axis=0)
        
        latent_vector = encoder.predict({
            "input_radical": test_rad_num,
            "input_sequence": test_seq_num
        }, verbose=0)
        
        predictions_dict = decoder.predict({"latent_input": latent_vector}, verbose=0)
        
        pred_rad_probs = np.squeeze(predictions_dict["output_radical"])
        pred_seq_probs = np.squeeze(predictions_dict["output_sequence"])
        
        pred_rad_idx = np.argmax(pred_rad_probs)
        reconstructed_rad = r_vocab[pred_rad_idx] if pred_rad_idx < len(r_vocab) else "UNK"
        
        reconstructed_seq_tokens = []
        for step in range(MAX_SEQ_LEN):
            token_idx = np.argmax(pred_seq_probs[step])
            token_str = c_vocab[token_idx] if token_idx < len(c_vocab) else "UNK"
            
            # Stop printing tokens if the model outputs a padding marker
            if token_str in ["", "PAD", "[PAD]"]:
                break
            reconstructed_seq_tokens.append(token_str)
            
        reconstructed_seq = "".join(reconstructed_seq_tokens) # Cleared spacing join boundaries
        
        print(f"Character: 【 {target_char} 】")
        print(f"  ├─ RADICAL  -> Original: {orig_rad:<5} | Reconstructed: {reconstructed_rad}")
        print(f"  └─ SEQUENCE -> Original: {orig_seq:<20} | Reconstructed: {reconstructed_seq}")
        print("-" * 60)

# Run Sample Validations
sample_reconstructions(
    encoder=encoder_model, decoder=decoder_model,
    raw_rads=raw_input_rads, raw_seqs=raw_input_seqs,
    num_rads=target_rads, num_seqs=target_seqs,
    char_labels=char_index,
    r_vocab=radical_lookup.get_vocabulary(), 
    c_vocab=sequence_vectorizer.get_vocabulary(),
    num_samples=5
)

# ----------------------------------------------------
# 8. EXTRACT AND SAVE ALL 10K DENSE EMBEDDINGS
# ----------------------------------------------------
print("\nExtracting final dense embedding vectors...")
final_embeddings_matrix = encoder_model.predict({
    "input_radical": target_rads, 
    "input_sequence": target_seqs
}, batch_size=256)

np.save("hanzi_embeddings_64d.npy", final_embeddings_matrix)

with open("hanzi_index_lookup.txt", "w", encoding="utf-8") as f:
    f.write(",".join(char_index.tolist()))

print(f"Success! Saved embedding weights matrix shape {final_embeddings_matrix.shape} to local storage.")



# --- SAVE ENCODER WEIGHTS ---
encoder_model.save_weights("hanzi_encoder_weights.weights.h5")

# --- SAVE DYNAMIC LAYER VOCABULARIES ---
# This ensures your lookup IDs never shift or change
np.save("saved_radical_vocab.npy", radical_lookup.get_vocabulary())
np.save("saved_component_vocab.npy", sequence_vectorizer.get_vocabulary())

print("Encoder assets successfully frozen and saved to disk!")

