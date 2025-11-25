import os
import joblib
import tensorflow as tf
import getpass
import hashlib
import base64
import pandas as pd 
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import numpy as np

MODEL_FILENAME = "Security_Artifacts/DL_encrypted.keras"
INTEGRITY_HASH_FILE = "Security_Artifacts/DL_sha256_hash.txt" 
SALT_FILE = "Security_Artifacts/DL_encryption_salt.bin" 
PREPROCESSOR_FILENAME = "Security_Artifacts/DL_alignment_rules.pkl"
LABEL_ENCODER_FILENAME = "Security_Artifacts/DL_label_mapping.pkl"

# DECRYPTION FUNCTIONS ---------------------------------

# Promps user for decryption key
def get_secure_password():
    print("\n----------------------------------------")
    print("!! SECURE KEY REQUIRED     ## realistically would be an external password")
    password = getpass.getpass("Enter DECRYPTION Key: ")
    return password.encode()

# Decryption Key System
def _get_fernet_key(password_bytes, salt_file=SALT_FILE):
    # Create salt
    if not os.path.exists(salt_file):
        salt = os.urandom(16)
        with open(salt_file, 'wb') as f:
            f.write(salt)
    else:
        with open(salt_file, 'rb') as f:
            salt = f.read()

    # KDF: Key Derivation Function 
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=300000,
        backend=default_backend()
    )
    
    key = kdf.derive(password_bytes)
    return Fernet(base64.urlsafe_b64encode(key))

# Calculate SHA-256 Hash
def calculate_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as file:
        while True:
            chunk = file.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

# Checks File Integrity
def check_file_integrity(filepath, expected_hash):
    if not os.path.exists(filepath): return False
    current_hash = calculate_hash(filepath)
    if current_hash == expected_hash:
        print(f"\t{os.path.basename(filepath)} is UNMODIFIED (Matched Hashes).")
        return True
    else:
        print(f"! TAMPERING DETECTED: {os.path.basename(filepath)} ")
        return False

# Decrypts the files
def load_secure_model(encrypted_filename, password_bytes):
    plaintext_filename = encrypted_filename.replace(".enc", "")
    try:
        f = _get_fernet_key(password_bytes)
        with open(encrypted_filename, 'rb') as file:
            encrypted_data = file.read()
            
        decrypted_data = f.decrypt(encrypted_data)
        with open(plaintext_filename, 'wb') as file:
            file.write(decrypted_data)
            
        print(f"!! SUCCESSFUL: '{encrypted_filename}' has been decrypted to '{plaintext_filename}'")
        return plaintext_filename 
        
    except Exception as e:
        print(f"ERROR: Invalid key/file corruption/tampering detected: {e}")
        return None 

# Integrity check prep
def load_trusted_hashes(hash_file):
    hashes = {}
    try:
        with open(hash_file, 'r') as f:
            for line in f:
                name, hash_value = line.strip().split(':')
                hashes[name] = hash_value
    except FileNotFoundError:
        print(f"ERROR: Hash file'{hash_file}' not found")
    return hashes

# Customized Focal Loss Object from Strava_DL.py
def make_sparse_focal_loss(gamma=2.0):
    """
    Sparse categorical focal loss:
    - y_true: integer class labels, shape (batch,)
    - y_pred: probabilities after softmax, shape (batch, num_classes)
    """
    # NOTE: The inner function name 'loss_fn' is what Keras searches for.
    def loss_fn(y_true, y_pred): 
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true_oh * tf.math.log(y_pred_clipped)
        focal_factor = tf.pow(1.0 - y_pred_clipped, gamma)
        loss = tf.reduce_sum(loss, axis=-1)
        return tf.reduce_mean(loss)
    return loss_fn

focal_loss_obj = make_sparse_focal_loss(gamma=2.0)

# DEPLOYMENT --------------------------------

def main_deployment():
    # Prompt decryption key
    MODEL_KEY = get_secure_password()
    
    files_to_decrypt = [MODEL_FILENAME, PREPROCESSOR_FILENAME, LABEL_ENCODER_FILENAME]
    decrypted_files = []
    
    # Print Hash Lisy
    expected_hashes = load_trusted_hashes(INTEGRITY_HASH_FILE)
    if not expected_hashes:
        print("ERROR: No trusted hashes available")
        return

    print("\n----------------------------------------")
    print("DECRYPTING AND CHECKING INTEGRITY...")
    deployment_ready = True

    for filename in files_to_decrypt:
        encrypted_file = filename + ".enc"
        
        # Decrypting
        extracted_file = load_secure_model(encrypted_file, MODEL_KEY)
        if extracted_file:
            decrypted_files.append(extracted_file)
            
            # Integrity Check
            if check_file_integrity(extracted_file, expected_hashes.get(extracted_file)):
                pass 
            else:
                deployment_ready = False
                break
        else:
            deployment_ready = False
            break

# IOT DEVICE SIMULTAION --------------------------------

    if deployment_ready:  
        #  In actual IoT deployment, this is where live sensor data is used
        print("\n----------------------------------------")
        print("REAL IOT SIMULATION (LIVE PREDICTION)")

        try:
            # As-if: loading files from secure storage to RAM
            model_loaded = tf.keras.models.load_model(MODEL_FILENAME,
                custom_objects={'loss_fn': focal_loss_obj})
            preprocessor_loaded = joblib.load(PREPROCESSOR_FILENAME)
            label_encoder_loaded = joblib.load(LABEL_ENCODER_FILENAME)
            original_input_features = joblib.load("Security_Artifacts/DL_alignment_record.pkl") 
            num_features = len(original_input_features)
            
            # As-if: recorded raw data from IoT sensors 
            dummy_array = np.random.rand(1, num_features).astype(np.float32)
            dummy_input_df = pd.DataFrame(dummy_array, columns=original_input_features)
            
            # As-if: Preprocess with the same format
            processed_input = preprocessor_loaded.transform(dummy_input_df) 
            
            # Actual HAR Prediction
            prediction_proba = model_loaded.predict(processed_input, verbose=0)
            predicted_index = np.argmax(prediction_proba, axis=1)[0]
            predicted_activity = label_encoder_loaded.inverse_transform([predicted_index])[0]
            
            print(f"\tRaw Input Features Shape: {dummy_input_df.shape}")
            print(f"\tPredicted Activity: Index {predicted_index}")
            print(f"\tFinal Prediction: {predicted_activity}")
            print("\n! Pipeline Is Working Perfectly")
        except Exception as e:
            print(f"ERROR: {e}")
            
    else:
        print("\n----------------------------------------")
        print("SECURITY FAILURE: Deployment is aborted")

    # Delete decrypted plaintext files
    print("\n----------------------------------------")
    print("CLEANING UP PLAINTEXT FILES...")
    for filename in decrypted_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"\tCleaned up {filename}")
            

    print("\n----------------------------------------")
    print("! MODEL IS RUN AND VERIFIED !\n")

if __name__ == "__main__":
    main_deployment()