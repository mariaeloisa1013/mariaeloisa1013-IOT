# IoT Data Security & Ethics: ML/DL Activity Recognition
Project Brief:
  This project explores the critical security and ethical challenges posed by massive data collection from Internet of Things (IoT) devices (e.g., Strava/wearables). We use Machine Learning (ML) and Deep Learning (FFNN) models to demonstrate the high utility of this data for Human Activity Recognition (HAR) and, consequently, the immense risk of misuse by surveillance actors.

The final deliverable includes a comparative performance analysis and a practical demonstration of cryptographic security measures applied directly to the trained AI assets.

# Objectives & Findings
- Objective	Proof Point	Security: Ethical Implication
- Model Comparison	DL (FFNN) achieved significantly higher F1-Scores than the ML baseline.	High Value: Confirms the data's utility for detailed profiling and surveillance.
- Ethical Analysis	Quantifiable 0.00 F1-Score on minority classes (e.g., 'Run').	Algorithmic Bias: Proves the ethical failure of centralized, imbalanced data collection.
- Asset Defense	Successful demonstration of Integrity Checks and AES Encryption on the final model files.	Necessity: Demonstrates the practical defenses required to protect the high-value 'black-box' AI asset from tampering.

# Setup and Installation
1. Environment Setup (Recommended)
    It is highly recommended to use a virtual environment (venv) to isolate project dependencies.
    - Create the virtual environment
    **python3 -m venv .venv**
    - Activate the virtual environment
    **source .venv/bin/activate**

2. Dependency Installation
The project requires core data science libraries, TensorFlow for Deep Learning, and specialized libraries for cryptographic security (pyzipper, cryptography).

Install all necessary packages in one command:
**pip install pandas numpy scikit-learn tensorflow joblib matplotlib seaborn pyzipper cryptography**


# Project Structure
```
SECURINGTHEIOT_AS1/
├── .venv/                              # Isolated Python environment for dependency management
|
├── Data_Preprocessing/                 # PHASE 1: Data Cleaning, Merging, and Initial Transformation
│   ├── Personal_Data/                  # Source folder for raw, anonymous personal data files
│   │   ├── MariaButaslac.csv           
│   │   ├── HyacinthToribio.csv        
│   │   └── SofiaBorcelo.csv            
│   ├── Public_Data/                    # Source folder for external, large-scale public data files
│   │   ├── public1.csv                
│   │   ├── public2.csv                 
│   │   ├── public3.csv                 
│   │   └── public4.csv                
│   ├── Clean_DateFormat.py             # Script to handle initial date/time string format inconsistencies
│   ├── Merge&PreProcessor.py           # Script to merge all raw data, clean, impute, and generate baseline artifacts
│   ├── MERGEDpersonaldataset.csv       # FINAL processed and scaled personal test set (Data Output)
│   └── MERGEDpublicdataset.csv         # FINAL processed and scaled public training set (Data Output)
|
├── Deep_Learning_Model/                # PHASE 2: Deep Learning Model Training and Security Implementation
│   ├── Deep_Learning.py                # Main DL model: Loads data, aligns features, trains 1D CNN, and implements security concepts
│   ├── SEC_verify_DL_integrity.py      # Script to verify the integrity of the SecuredData.csv (uses HMAC key)
|
├── Machine_Learning_Model/             # PHASE 3: Traditional ML Model Training and Core Security Functions
│   ├── __pycache__/                    
│   ├── Machine_Learning.py             # Main ML model: Trains Random Forest on public data and runs integrity checks
│   ├── SEC_library.py                  # Core secuurity script: Contains HMAC generation, Fernet encryption/decryption functions, Key Derivation Functions
│   └── SEC_verify_ML_integrity.py      # Script to verify the integrity of the Traditional ML model artifacts 
|
├── Security_Artifacts/                 # PHASE 4: Repository for all ENCRYPTED model logic and audit records
│   ├── DL_alignment_record.pkl         # Final list of common features used in the DL model
│   ├── DL_alignment_rules.pkl.enc      # DL Preprocessor (Master Scaling/OHE Rules)
│   ├── DL_encrypted.keras.enc          # Deep Learning Model (The Prediction Engine/IP)
│   ├── DL_encryption_salt.bin          # Salt file used for key derivation for encryption
│   ├── DL_label_mapping.pkl.enc        # Label Encoder (Activity Dictionary)
│   ├── DL_sha256_hash.txt              # SHA-256 HASH of the unencrypted DL Model
|   |
│   ├── DP_activity_label_encoder.pkl   # Baseline Label Encoder from initial data fit (Preprocessing Artifact)
│   └── DP_main_preprocessor.pkl        # Baseline Preprocessor from initial data fit (Preprocessing Artifact)
|   |
|   └── ML_encrypted_results.csv        # Encrypted ML prediction file (Contains Encrypted ID and Integrity MAC)                  
```


# Execution Guide

Phase 1: Data Preprocessing (Creating the Final Dataset)
  This phase cleans the raw data, applies feature engineering, and saves the final preprocessed files required by the models.

  1. Activate Environment:
    - Navigate to the Data_Preprocessing folder.
    - Activate its local environment:
      - macOS/Linux: **source .venv/bin/activate**
      - Windows (CMD): **.\.venv\Scripts\activate.bat**
  2. Run Cleaning Scripts: Execute the individual cleaning scripts. These scripts read the raw CSVs, apply transformations (imputation, scaling), and save the final clean CSVs.
    - **python Data_Preprocessing/PreProcessor.py**
  3. Verify Outputs: Confirm that the following files have been created in the Data_Preprocessing directory:
    - MERGEDpersonaldataset.csv / MERGEDpublicdataset.csv
    - Security_Artifacts/DP_activity_label_encoder.pkl
    - Security_Artifacts/DP_main_preprocessor.pkl (The scaling/encoding rules)
  4. Deactivate Environment: **deactivate**

Phase 2: Machine Learning Model (Baseline Comparison)
  This phase trains the simpler ML model for performance comparison (often using the same data and preprocessor rules).
  1. Activate Environment:
    - Navigate to the Machine_Learning_Model folder.
    - Activate its local environment:
      - macOS/Linux: **source .venv/bin/activate**
      - Windows (CMD): **.\.venv\Scripts\activate.bat**
  2. Run ML Script: Execute the ML training script.
      - **python Machine_Learning.py**
  3. Verify Integrity: Run the integrity script to demonstrate security principles on the ML model's output (or data).
    - **python SEC_verify_ML_integrity**
  4. Deactivate Environment: **deactivate**

Phase 3: Deep Learning Model (Training and Security Demonstration)
  This phase uses the processed data to train the FFNN model, establishes the trusted hash, encrypts the assets, and verifies integrity.

  1. Activate Environment:
    - Navigate to the Deep_Learning_Model folder.
    - Activate its local environment:
      - macOS/Linux: **source .venv/bin/activate**
      - Windows (CMD): **.\.venv\Scripts\activate.bat**
  2. Run Main Model Script: Execute the DL script. This script will perform the full cycle (train, save, hash, encrypt, delete plaintext, decrypt, and verify).
    - **python Deep_Learning.py**
  3. Verify Security Outputs: Confirm the script has generated the following encrypted files (and deleted their plaintext .pkl / .keras counterparts):
    - Security_Artifacts/DL_encrypted.keras.enc
    - Security_Artifacts/DL_alignment_rules.pkl.enc
    - Security_Artifacts/DL_label_mapping.pkl.enc
    - Security_Artifacts/DL_sha256_hash.txt (Contains the trusted fingerprint)
  4. Test Verification (Optional): Run the verification script to ensure the security functions work independently.
    - **python SEC_verify_DL_integrity.py**
  5. Deactivate Environment: **deactivate**



---------------------------------
This project was developed by **Maria Eloisa Butaslac, Hyacinth Ava Toribio, Sofia Lorin Borcelo** [CS Y3: GROUP 4]
