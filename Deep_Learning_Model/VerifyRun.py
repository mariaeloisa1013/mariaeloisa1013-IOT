import os
import joblib
import tensorflow as tf
import getpass
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import numpy as np # Needed for array manipulation

# --- CONFIGURATION (Must match the training script!) ---
MODEL_FILENAME = "strava_activity_classify.keras"
PREPROCESSOR_FILENAME = "preprocessor.pkl"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"
INTEGRITY_HASH_FILE = "model_integrity_hash.txt" 
SALT_FILE = "encryption_salt.bin" 
# --- END CONFIGURATION ---


# --- SECURITY DECRYPTION FUNCTIONS (LOAD/VERIFY) ---

def get_secure_password():
    """Prompts the user for the decryption key securely."""
    print("\n--- SECURE KEY INPUT REQUIRED (DECRYPTION) ---")
    password = getpass.getpass("Enter Decryption Key: ")
    print("-----------------------------------")
    return password.encode()


def _get_fernet_key(password_bytes, salt_file=SALT_FILE):
    """Derives a strong encryption key from the password and a stored salt."""
    if not os.path.exists(salt_file):
        raise FileNotFoundError(f"Salt file not found: {salt_file}. Cannot derive key.")
        
    with open(salt_file, 'rb') as f:
        salt = f.read()

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=300000,
        backend=default_backend()
    )
    key = kdf.derive(password_bytes)
    return Fernet(base64.urlsafe_b64encode(key))

def calculate_hash(filepath):
    """Calculates the SHA-256 hash (digital fingerprint) of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as file:
        while True:
            chunk = file.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def check_file_integrity(filepath, expected_hash):
    """Checks file integrity against a stored hash."""
    if not os.path.exists(filepath): return False
    current_hash = calculate_hash(filepath)
    if current_hash == expected_hash:
        print(f"✅ INTEGRITY CHECK: {os.path.basename(filepath)} is UNMODIFIED (Hash Match).")
        return True
    else:
        print(f"❌ INTEGRITY CHECK: {os.path.basename(filepath)} has been TAMPERED with!")
        return False

def load_secure_model(encrypted_filename, password_bytes):
    """Decrypts the file and writes the plaintext for loading and verification."""
    plaintext_filename = encrypted_filename.replace(".enc", "")
    try:
        f = _get_fernet_key(password_bytes)
        
        with open(encrypted_filename, 'rb') as file:
            encrypted_data = file.read()
            
        decrypted_data = f.decrypt(encrypted_data)
        
        with open(plaintext_filename, 'wb') as file:
            file.write(decrypted_data)
            
        print(f"✅ DECRYPTION SUCCESSFUL: '{encrypted_filename}' extracted to '{plaintext_filename}'.")
        return plaintext_filename 
        
    except Exception as e:
        print(f"❌ DECRYPTION FAILED. Invalid key, file corruption, or tampering detected: {e}")
        return None 

def load_trusted_hashes(hash_file):
    """Loads trusted hashes from the integrity file."""
    hashes = {}
    try:
        with open(hash_file, 'r') as f:
            for line in f:
                name, hash_value = line.strip().split(':')
                hashes[name] = hash_value
    except FileNotFoundError:
        print(f"❌ ERROR: Trusted hash file '{hash_file}' not found. Cannot verify integrity.")
    return hashes

# --- END SECURITY DECRYPTION FUNCTIONS ---


def main_deployment():
    # 0. GET SECURE PASSWORD
    MODEL_KEY = get_secure_password()
    
    files_to_decrypt = [MODEL_FILENAME, PREPROCESSOR_FILENAME, LABEL_ENCODER_FILENAME]
    decrypted_files = []
    
    # Load trusted hash list
    expected_hashes = load_trusted_hashes(INTEGRITY_HASH_FILE)
    if not expected_hashes:
        print("ABORTING DEPLOYMENT: No trusted hashes available.")
        return

    print("\n--- BEGIN DECRYPTION AND INTEGRITY CHECK ---")
    deployment_ready = True

    for filename in files_to_decrypt:
        encrypted_file = filename + ".enc"
        
        # 1. Decrypt the file
        extracted_file = load_secure_model(encrypted_file, MODEL_KEY)
        if extracted_file:
            decrypted_files.append(extracted_file)
            
            # 2. Check integrity
            if check_file_integrity(extracted_file, expected_hashes.get(extracted_file)):
                pass 
            else:
                deployment_ready = False
                break
        else:
            deployment_ready = False
            break

    # Final deployment check
    if deployment_ready:
        print("\n✅ DEPLOYMENT SUCCESS: Loading secure components...")
        
        try:
            # 3. Load the verified components into memory
            model_loaded = tf.keras.models.load_model(MODEL_FILENAME)
            preprocessor_loaded = joblib.load(PREPROCESSOR_FILENAME)
            label_encoder_loaded = joblib.load(LABEL_ENCODER_FILENAME)
            
            print("Model and preprocessors loaded successfully. READY FOR INFERENCE.")
            
            # --- SIMULATION OF REAL-TIME INFERENCE ---
            # NOTE: In a real IoT deployment, this is where live sensor data would be used.
            print("\n--- SIMULATING INFERENCE (LIVE PREDICTION) ---")
            
            # Create dummy input data (must have the same 26 features as training)
            # Replace this with real IoT sensor readings in production
            dummy_input = np.random.rand(1, preprocessor_loaded.n_features_in_).astype(np.float32)
            
            # Preprocess the dummy data using the verified preprocessor
            processed_input = preprocessor_loaded.transform(dummy_input)
            
            # Make a prediction
            prediction_proba = model_loaded.predict(processed_input, verbose=0)
            predicted_index = np.argmax(prediction_proba, axis=1)[0]
            predicted_activity = label_encoder_loaded.inverse_transform([predicted_index])[0]
            
            print(f"RAW INPUT FEATURES SHAPE: {dummy_input.shape}")
            print(f"PREDICTED ACTIVITY (Raw Output): Index {predicted_index}")
            print(f"FINAL PREDICTION: **{predicted_activity}**")
            
        except Exception as e:
            print(f"❌ RUNTIME ERROR: Failed to load/run verified model. {e}")
            
    else:
        print("\nSECURITY FAILURE: Deployment aborted.")

    # Clean up all decrypted plaintext files
    print("\n--- Cleaning up temporary plaintext files ---")
    for filename in decrypted_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Cleaned up {filename}.")
            
    print("\nVerification process complete.")

if __name__ == "__main__":
    main_deployment()
    