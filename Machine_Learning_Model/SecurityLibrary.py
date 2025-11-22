import hmac
import hashlib
import os
from cryptography.fernet import Fernet
import pandas as pd


# HMAC Key:  for data integrity
HMAC_SECRET_KEY = os.urandom(32) 

# Fernet Key: for encryption/decryption 
FERNET_KEY = Fernet.generate_key()
CIPHER = Fernet(FERNET_KEY)


# DERIVED FUNCTIONS -----------------------------

# Generates an HMAC-SHA256 for a series of data points.
def generate_hmac(data_series: pd.Series, key: bytes = HMAC_SECRET_KEY) -> str:
    data_str = data_series.astype(str).str.cat(sep='|')
    data_bytes = data_str.encode('utf-8')
    
    hm = hmac.new(key, data_bytes, hashlib.sha256)
    return hm.hexdigest()

#Encrypts String ID with the Fernet 
def encrypt_id(activity_id: str) -> bytes:
    return CIPHER.encrypt(activity_id.encode())

def get_fernet_key() -> bytes: # for decryption
    return FERNET_KEY

def get_hmac_key() -> bytes: # for logging/sharing
    return HMAC_SECRET_KEY