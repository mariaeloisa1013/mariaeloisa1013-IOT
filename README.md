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
pip install pandas numpy scikit-learn tensorflow joblib matplotlib seaborn pyzipper cryptography getpass

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
To replicate the security demonstration and model training:

1. Ensure data is ready: Make sure your final, clean CSV (strava_processed_data.csv) is generated by the preprocessing script.
2. Run the Main DL Script: This script handles model training, integrity checking, encryption, and final evaluation.
3. Run (Deep_Learning_Model/VerifyRun.py)

Expected Output Log Flow
    1. Training Metrics (Accuracy/F1-Score)
    2. INTEGRITY CHECK: strava_activity_classify.keras is UNMODIFIED (Hash Match).
    3. User is prompted to enter secure key.
    4. CRYPTOGRAPHY APPLIED: Model encrypted and stored as [file.enc]. Plaintext removed.
    5. DECRYPTION SUCCESSFUL: Model extracted for loading.
    6. INTEGRITY CHECK: strava_activity_classify.keras is UNMODIFIED (Hash Match).
    7. DEPLOYMENT SUCCESS: All three model components loaded successfully after security checks.


---------------------------------
This project was developed by Maria Eloisa Butaslac, Hyacinth Ava Toribio, Sofia Lorin Borcelo [CS Y3 WKND: GROUP 4]
