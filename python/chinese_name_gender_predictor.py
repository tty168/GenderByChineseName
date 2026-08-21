import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import random

# ----------------------------------------------------
# 1. LOAD SAVED ASSETS & INFERENCE VOCABULARIES
# ----------------------------------------------------
saved_rad_vocab = np.load("saved_radical_vocab.npy", allow_pickle=True).tolist()
saved_comp_vocab = np.load("saved_component_vocab.npy", allow_pickle=True).tolist()

MAX_SEQ_LEN = 12
LATENT_DIM = 64

radical_lookup = layers.StringLookup(vocabulary=saved_rad_vocab, output_mode="int", oov_token="UNK")

def split_by_individual_characters(input_string):
    return tf.strings.unicode_split(input_string, input_encoding="UTF-8")

sequence_vectorizer = layers.TextVectorization(
    standardize=None, split=split_by_individual_characters, 
    output_mode="int", output_sequence_length=MAX_SEQ_LEN, vocabulary=saved_comp_vocab
)

num_radicals = radical_lookup.vocabulary_size()
num_components = sequence_vectorizer.vocabulary_size()

# ----------------------------------------------------
# 2. REBUILD THE EXACT PRE-TRAINED ENCODER PATTERN
# ----------------------------------------------------
enc_rad_in = layers.Input(shape=(1, 1), dtype=tf.int32, name="enc_rad_idx")
enc_seq_in = layers.Input(shape=(1, MAX_SEQ_LEN), dtype=tf.int32, name="enc_seq_idx")

# Keras 3 compatible dimension collapse layer
rad_squeezed = layers.Reshape((1,))(enc_rad_in)          
seq_squeezed = layers.Reshape((MAX_SEQ_LEN,))(enc_seq_in) 

rad_embed = layers.Embedding(input_dim=num_radicals, output_dim=32)(rad_squeezed)
rad_features = layers.Flatten()(rad_embed)

seq_embed = layers.Embedding(input_dim=num_components, output_dim=64)(seq_squeezed)
seq_conv = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(seq_embed)
seq_features = layers.GlobalAveragePooling1D()(seq_conv)

fused_features = layers.Concatenate()([rad_features, seq_features])
latent_embedding = layers.Dense(LATENT_DIM, activation=None, name="hanzi_latent_space")(fused_features)

base_encoder = Model(inputs=[enc_rad_in, enc_seq_in], outputs=latent_embedding, name="Pretrained_Hanzi_Encoder")
base_encoder.load_weights("hanzi_encoder_weights.weights.h5")
base_encoder.trainable = False 

# ----------------------------------------------------
# 3. BUILD THE MULTI-CHARACTER GENDER CLASSIFIER 
# ----------------------------------------------------
char1_rad = layers.Input(shape=(1, 1), dtype=tf.int32, name="char1_rad")
char1_seq = layers.Input(shape=(1, MAX_SEQ_LEN), dtype=tf.int32, name="char1_seq")

char2_rad = layers.Input(shape=(1, 1), dtype=tf.int32, name="char2_rad")
char2_seq = layers.Input(shape=(1, MAX_SEQ_LEN), dtype=tf.int32, name="char2_seq")

char1_embedding = base_encoder([char1_rad, char1_seq])
char2_embedding = base_encoder([char2_rad, char2_seq])

# Side-by-side concatenation maps position semantic logic flawlessly
name_representation = layers.Concatenate()([char1_embedding, char2_embedding]) # (None, 128)

dense_1 = layers.Dense(128, activation="relu")(name_representation)
dropout_1 = layers.Dropout(0.3)(dense_1)
dense_2 = layers.Dense(64, activation="relu")(dropout_1)
dropout_2 = layers.Dropout(0.2)(dense_2)
gender_output = layers.Dense(1, activation="sigmoid", name="gender_probability_ratio")(dropout_2)

gender_model = Model(
    inputs=[char1_rad, char1_seq, char2_rad, char2_seq], 
    outputs=gender_output, 
    name="Name_Gender_Predictor"
)
gender_model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])
gender_model.summary()

# ----------------------------------------------------
# 4. LOAD CHARACTER STRUCTURAL LOOKUPS & PRE-COMPILE CACHE
# ----------------------------------------------------
def load_sqlite_export(file_path, separator="|"):
    mapping = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(separator)
            if len(parts) == 3:
                char, rad, decomp = parts
                mapping[char.strip()] = (rad.strip(), decomp.strip())
    return mapping

db_lookup = load_sqlite_export("training_dataset.txt", separator="|")

print("Pre-compiling database character vector map cache...")
char_vector_cache = {}
for char, (rad_str, seq_str) in db_lookup.items():
    r_id = radical_lookup(np.array([[rad_str]])).numpy().astype(np.int32).reshape(1, 1)
    s_ids = sequence_vectorizer(np.array([[seq_str]])).numpy().astype(np.int32).reshape(1, MAX_SEQ_LEN)
    char_vector_cache[char] = (r_id, s_ids)

# Standardized fallback shapes for unknown characters and padding items
fallback_rad = np.array([[1]], dtype=np.int32)       # UNK Index Shape: (1, 1)
fallback_seq = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32) # PAD sequence vectors Shape: (1, 12)

pad_rad = np.array([[0]], dtype=np.int32)             # PAD Index Shape: (1, 1)
pad_seq = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)     # PAD sequence vectors Shape: (1, 12)

# ----------------------------------------------------
# 5. STREAM DATA GENERATOR PIPELINE (Tab Separated Dataverse)
# ----------------------------------------------------
def gender_dataset_generator(file_path, delimiter="\t"):
    with open(file_path, "r", encoding="utf-8") as f:
        f.readline() # Skip header row
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(delimiter)
            if len(parts) < 4: continue
            
            name = parts[0].strip()
            try:
                ratio = float(parts[3].strip())
            except ValueError:
                continue
            
            chars = [c for c in name][:2]
            if not chars: continue
            
            # Extract Character 1 features
            c1_r, c1_s = char_vector_cache.get(chars[0], (fallback_rad, fallback_seq))
            
            # Extract Character 2 features
            if len(chars) == 2:
                c2_r, c2_s = char_vector_cache.get(chars[1], (fallback_rad, fallback_seq))
            else:
                c2_r, c2_s = pad_rad, pad_seq
                
            yield (
                {
                    "char1_rad": c1_r, "char1_seq": c1_s,
                    "char2_rad": c2_r, "char2_seq": c2_s
                }, 
                ratio
            )

# ----------------------------------------------------
# 6. INITIALIZE STREAMING ITERATOR DATASET
# ----------------------------------------------------
#DATASET_PATH = "gender/dataverse_files/CnGender.txt"
DATASET_TRAIN_PATH = "gender/dataverse_files/CnGender_train.txt"
DATASET_VALID_PATH = "gender/dataverse_files/CnGender_valid.txt"

large_dataset = tf.data.Dataset.from_generator(
    lambda: gender_dataset_generator(DATASET_TRAIN_PATH, delimiter="\t"),
    output_signature=(
        {
            "char1_rad": tf.TensorSpec(shape=(1, 1), dtype=tf.int32),
            "char1_seq": tf.TensorSpec(shape=(1, MAX_SEQ_LEN), dtype=tf.int32),
            "char2_rad": tf.TensorSpec(shape=(1, 1), dtype=tf.int32),
            "char2_seq": tf.TensorSpec(shape=(1, MAX_SEQ_LEN), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
)

BATCH_SIZE = 512
train_dataset = large_dataset.shuffle(buffer_size=20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Starting pipeline training loops over locked transfer weights...")
gender_model.fit(train_dataset, epochs=20, shuffle=None)

# ----------------------------------------------------
# 7. INFERENCE EVALUATION PROFILER
# ----------------------------------------------------
def predict_gender(target_name):
    chars = [c for c in target_name][:2]
    c1_r, c1_s = char_vector_cache.get(chars[0], (fallback_rad, fallback_seq))
    
    if len(chars) == 2:
        c2_r, c2_s = char_vector_cache.get(chars[1], (fallback_rad, fallback_seq))
    else:
        c2_r, c2_s = pad_rad, pad_seq
        
    male_prob = gender_model.predict({
        "char1_rad": np.expand_dims(c1_r, axis=0), "char1_seq": np.expand_dims(c1_s, axis=0),
        "char2_rad": np.expand_dims(c2_r, axis=0), "char2_seq": np.expand_dims(c2_s, axis=0)
    }, verbose=0)
    
    male_prob = float(np.squeeze(male_prob))
    print(f"Name: {target_name} -> Male Probability: {male_prob * 100:.2f}% ({'Male 男' if male_prob > 0.5 else 'Female 女'})")
    print("-" * 50)

print("\n" + "="*50)
print("      EVALUATING CUSTOM PROFILING SYSTEM INFERENCE")
print("="*50)
predict_gender("建民")
predict_gender("云霞")
predict_gender("新")


# ----------------------------------------------------
# 8. PRODUCTION MODEL EXPORT
# ----------------------------------------------------
print("\n" + "="*60)
print("      EXPORTING TRAINED GENDER PREDICTOR FOR PRODUCTION")
print("="*60)

# Save the entire fine-tuned neural network bundle natively in Keras 3 format
production_model_filename = "hanzi_name_gender_predictor.keras"
gender_model.save(production_model_filename)

print(f"Success! Model graph architecture and fine-tuned weights saved.")
print(f"Production Deployment Asset File: '{production_model_filename}'")
print("-" * 60)


# ----------------------------------------------------
# 9. LARGE-SCALE AUTOMATED EVALUATION PIPELINE (AUC & ACCURACY)
# ----------------------------------------------------
print("\n" + "="*60)
print("      COMPUTING BINARY CLASSIFICATION METRICS (AUC & ACCURACY)")
print("="*60)

def compute_auc_numpy(y_true_binary, y_pred_probabilities):
    """
    Computes the exact Area Under the ROC Curve (AUC-ROC) natively via the 
    Wilcoxon-Mann-Whitney statistic formula to dodge heavy scikit-learn imports.
    """
    pos_scores = y_pred_probabilities[y_true_binary == 1]
    neg_scores = y_pred_probabilities[y_true_binary == 0]
    
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0
        
    # Broadcast element-wise matching comparisons across matrices
    favorable_matches = pos_scores[:, None] > neg_scores[None, :]
    exact_ties = pos_scores[:, None] == neg_scores[None, :]
    
    auc_score = (np.sum(favorable_matches) + 0.5 * np.sum(exact_ties)) / (len(pos_scores) * len(neg_scores))
    return auc_score

def evaluate_production_model(file_path, model_pipeline, delimiter="\t", max_eval_records=20000):
    """
    Streams validation name slices from disk, extracts classification probabilities,
    binarizes the tracking shapes at a 0.5 threshold, and calculates Accuracy and AUC.
    """
    print(f"Streaming up to {max_eval_records} validation name records for performance scoring...")
    
    generator_stream = gender_dataset_generator(file_path, delimiter=delimiter)
    
    true_ratios = []
    predicted_probabilities = []
    
    for count, (inputs, true_ratio) in enumerate(generator_stream, 1):
        if count > max_eval_records:
            break
            
        # Add required batch axes
        c1_r = np.expand_dims(inputs["char1_rad"], axis=0)
        c1_s = np.expand_dims(inputs["char1_seq"], axis=0)
        c2_r = np.expand_dims(inputs["char2_rad"], axis=0)
        c2_s = np.expand_dims(inputs["char2_seq"], axis=0)
        
        # Run local forward inference evaluation pass
        pred_prob = model_pipeline.predict({
            "char1_rad": c1_r, "char1_seq": c1_s,
            "char2_rad": c2_r, "char2_seq": c2_s
        }, verbose=0)
        
        true_ratios.append(true_ratio)
        predicted_probabilities.append(float(np.squeeze(pred_prob)))
        
        if count % 5000 == 0:
            print(f"  Processed {count}/{max_eval_records} records...")

    # Convert continuous inputs over to structured numpy vector sets
    y_true_continuous = np.array(true_ratios, dtype=np.float32)
    y_pred_probs = np.array(predicted_probabilities, dtype=np.float32)
    
    # FIXED: Binarize continuous values at a standard 0.5 decision threshold
    y_true_binary = (y_true_continuous > 0.5).astype(np.int32)
    y_pred_binary = (y_pred_probs > 0.5).astype(np.int32)
    
    # Calculate performance metrics
    accuracy = np.mean(y_true_binary == y_pred_binary)
    auc_roc = compute_auc_numpy(y_true_binary, y_pred_probs)
    
    # Calculate underlying balance stats to ensure metrics are valid
    total_positives = np.sum(y_true_binary)
    total_negatives = len(y_true_binary) - total_positives
    
    print("\n" + "─"*50)
    print("      GLOBAL CLASSIFICATION EVALUATION METRICS")
    print("─"*50)
    print(f"  Total Validation Samples   : {len(y_true_binary)} (Male: {total_positives} | Female: {total_negatives})")
    print(f"  Classification Accuracy    : {accuracy * 100:.2f}%")
    print(f"  ROC Area Under Curve (AUC) : {auc_roc:.4f}")
    print("─"*50)

# Run verification evaluation over an unseen test slice of 20,000 dataverse lines
evaluate_production_model(
    file_path=DATASET_VALID_PATH,
    model_pipeline=gender_model,
    delimiter="\t",
    max_eval_records=20000
)

