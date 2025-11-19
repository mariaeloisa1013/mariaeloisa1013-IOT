import hmac
import hashlib
import os
from cryptography.fernet import Fernet
import pandas as pd # Needed for handling pandas Series in HMAC

# =================================================================
# 1. KEY MANAGEMENT: Secrets are loaded from the environment 
#    or generated as temporary tokens. NEVER hardcoded.
# =================================================================

# HMAC Key: Used for data integrity/authenticity (32 bytes)
# In production, this would be loaded from a secure vault.
HMAC_SECRET_KEY = os.urandom(32) 

# Fernet Key: Used for encryption/decryption (32 URL-safe bytes)
# Using a generated key for demonstration.
FERNET_KEY = Fernet.generate_key()
CIPHER = Fernet(FERNET_KEY)

# =================================================================
# 2. SECURITY FUNCTIONS
# =================================================================

def generate_hmac(data_series: pd.Series, key: bytes = HMAC_SECRET_KEY) -> str:
    """Generates an HMAC-SHA256 for a series of data points."""
    # Concatenate all data points into a single byte string
    data_str = data_series.astype(str).str.cat(sep='|')
    data_bytes = data_str.encode('utf-8')
    
    # Calculate and return the HMAC digest (hexadecimal string)
    hm = hmac.new(key, data_bytes, hashlib.sha256)
    return hm.hexdigest()

def encrypt_id(activity_id: str) -> bytes:
    """Encrypts a string ID using the Fernet cipher (Data Masking)."""
    return CIPHER.encrypt(activity_id.encode())

def get_fernet_key() -> bytes:
    """Returns the Fernet key. Crucial for decryption on the receiving end."""
    return FERNET_KEY

def get_hmac_key() -> bytes:
    """Returns the HMAC secret key for logging/sharing."""
    return HMAC_SECRET_KEY