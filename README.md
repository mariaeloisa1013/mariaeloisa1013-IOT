# IoT Data Security & Ethics: ML/DL Activity Recognition
Project Brief:
  This project explores the critical security and ethical challenges posed by massive data collection from Internet of Things (IoT) devices (e.g., Strava/wearables). We use Machine Learning (ML) and Deep Learning (FFNN) models to demonstrate the high utility of this data for Human Activity Recognition (HAR) and, consequently, the immense risk of misuse by surveillance actors.

The final deliverable includes a comparative performance analysis and a practical demonstration of cryptographic security measures applied directly to the trained AI assets.

# Objectives & Findings
- Objective	Proof Point	Security/Ethical Implication
- Model Comparison	DL (FFNN) achieved significantly higher F1-Scores than the ML baseline.	High Value: Confirms the data's utility for detailed profiling and surveillance.
- Ethical Analysis	Quantifiable 0.00 F1-Score on minority classes (e.g., 'Workout').	Algorithmic Bias: Proves the ethical failure of centralized, imbalanced data collection.
- Asset Defense	Successful demonstration of Integrity Checks and AES Encryption on the final model files.	Necessity: Demonstrates the practical defenses required to protect the high-value 'black-box' AI asset from tampering.

# Setup and Installation
1. Environment Setup (Recommended)
    It is highly recommended to use a virtual environment (venv) to isolate project dependencies.
    - Create the virtual environment
    python3 -m venv .venv
    - Activate the virtual environment
    source .venv/bin/activate

2. Dependency Installation
The project requires core data science libraries, TensorFlow for Deep Learning, and specialized libraries for cryptographic security (pyzipper, cryptography).

Install all necessary packages in one command:
'''pip install pandas numpy scikit-learn tensorflow joblib matplotlib seaborn pyzipper cryptography getpass

# Project Structure
```
SECURINGTHEIOT_AS1/
├── Data_Preprocessing/                   # PHASE 1: Data Cleaning, Feature Engineering, and Anonymization
│   ├── .venv/                            # Python Environment (for each folder, since this was worked in sections by multiple members)
|   |
│   ├── Personal_Data/                    # Folder for personal/anonymous raw data
│   │   ├── anon1.csv                     # Raw Anonymous Data Source 1
│   │   ├── anon2.csv                     # Raw Anonymous Data Source 2
│   │   ├── anon3.csv                     # Raw Anonymous Data Source 3
│   │   ├── Clean_DateFormat.py           # Script to standardize date/time formats
│   │   └── PSDataset_Cleaning.py         # Script for merging, cleaning, and feature engineering Personal Data
|   |
│   ├── Public_Data/                      # Folder for publicly sourced raw data
│   │   ├── PBDataset_Cleaning.py         # Script for merging, cleaning, and feature engineering Public Data
│   │   ├── public1.csv                   # Raw Public Data Source 1
│   │   ├── public2.csv                   # Raw Public Data Source 2
│   │   ├── public3.csv                   # Raw Public Data Source 3
│   │   └── public4.csv                   # Raw Public Data Source 4
│   ├── FINALPersonalDataset.csv          # Final output: Cleaned and processed Personal Data CSV
│   ├── FINALPublicDataset.csv            # Final output: Cleaned and processed Public Data CSV
│   ├── LabelEncoder.pkl                  # Saved object for decoding Activity Type (used by models)
│   └── PreProcessor.pkl                  # Saved feature scaling/encoding rules (StandardScaler, OHE)
|
├── Deep_Learning_Model/                  # PHASE 2: DL Model Training, Security Hardening, and Verification
│   ├── .venv/                            
│   ├── encryption_salt.bin               # Unique salt used for key derivation (part of symmetric encryption)
│   ├── label_encoder.pkl.enc             # Encrypted copy of the LabelEncoder object (Confidentiality At Rest)
│   ├── model_integrity_hash.txt          # File storing the trusted SHA-256 hash (Defense against Tampering)
│   ├── preprocessor.pkl.enc              # Encrypted copy of the preprocessing rules (Confidentiality At Rest)
│   ├── strava_activity_classify.keras.enc# Encrypted trained FFNN model file (The high-value AI asset)
│   ├── Strava_DL.py                      # Main script for training the FFNN model and applying security demos
│   └── VerifyRun.py                      # Script to test security functions independently
|
├── Machine_Learning_Model/               # PHASE 3: ML Model Training and Verification (Baseline Comparison)
│   ├── __pycache__/                      
│   ├── .venv/                            
│   ├── SecuredData.csv                   # Data file specific to the ML Model (optional, but often used)
│   ├── SecurityLibrary.py                # Library containing security functions (hashing, encryption, etc.)
│   ├── Strava_ML.py                      # Main script for training the ML baseline model (e.g., Random Forest)
│   └── VerifyIntegrity.py                # Script to test integrity check on the ML model/data
├── .gitattributes                    
└── README.md                             
```


# Execution Guide

Phase 1: Data Preprocessing (Creating the Final Dataset)
  This phase cleans the raw data, applies feature engineering, and saves the final preprocessed files required by the models.

  1.Activate Environment:
    - Navigate to the Data_Preprocessing folder.
    - Activate its local environment:
      - macOS/Linux: source .venv/bin/activate
      - Windows (CMD): .\.venv\Scripts\activate.bat
  2. Run Cleaning Scripts: Execute the individual cleaning scripts. These scripts read the raw CSVs, apply transformations (imputation, scaling), and save the final clean CSVs.
    - Personal Data: python Personal_Data/PSDataset_Cleaning.py
    - Public Data: python Public_Data/PBDataset_Cleaning.py
  3. Verify Outputs: Confirm that the following files have been created in the Data_Preprocessing directory:
    - FINALPersonalDataset.csv / FINALPublicDataset.csv
    - LabelEncoder.pkl
    - PreProcessor.pkl (The scaling/encoding rules)
  4. Deactivate Environment: deactivate

Phase 2: Machine Learning Model (Baseline Comparison)
  This phase trains the simpler ML model for performance comparison (often using the same data and preprocessor rules).
  1. Activate Environment:
    - Navigate to the Machine_Learning_Model folder.
    - Activate its local environment:
      - macOS/Linux: source .venv/bin/activate
      - Windows (CMD): .\.venv\Scripts\activate.bat
  2. Run ML Script: Execute the ML training script.
      - python Strava_ML.py
  3. Verify Integrity: Run the integrity script to demonstrate security principles on the ML model's output (or data).
    - python VerifyIntegrity.py
  4. Deactivate Environment: deactivate

Phase 3: Deep Learning Model (Training and Security Demonstration)
  This phase uses the processed data to train the FFNN model, establishes the trusted hash, encrypts the assets, and verifies integrity.

  1. Activate Environment:
    - Navigate to the Deep_Learning_Model folder.
    - Activate its local environment:
      - macOS/Linux: source .venv/bin/activate
      - Windows (CMD): .\.venv\Scripts\activate.bat
  2. Run Main Model Script: Execute the DL script. This script will perform the full cycle (train, save, hash, encrypt, delete plaintext, decrypt, and verify).
    - python Strava_DL.py
  3. Verify Security Outputs: Confirm the script has generated the following encrypted files (and deleted their plaintext .pkl / .keras counterparts):
    - strava_activity_classify.keras.enc
    - label_encoder.pkl.enc
    - preprocessor.pkl.enc
    - model_integrity_hash.txt (Contains the trusted fingerprint)
  4. Test Verification (Optional): Run the verification script to ensure the security functions work independently.
    - python VerifyRun.py
  5. Deactivate Environment: deactivate



---------------------------------
This project was developed by Maria Eloisa Butaslac, Hyacinth Ava Toribio, Sofia Lorin Borcelo [CS Y3 WKND: GROUP 4]
