import pandas as pd
import hmac
import hashlib
import os


def generate_hmac(data_series: pd.Series, key: bytes) -> str:
    data_str = data_series.astype(str).str.cat(sep='|')
    data_bytes = data_str.encode('utf-8')
    hm = hmac.new(key, data_bytes, hashlib.sha256)
    return hm.hexdigest()

def verify_data_integrity(file_path, hmac_key_bytes):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: '{file_path}' not found")
        return

    tamper_detected = 0
    
    for index, row in df.iterrows():
        data_to_seal = pd.Series([
            row['Encrypted ID'], 
            row['Activity Type'], 
            row['Predicted Activity Type']
        ])
        
        new_mac = generate_hmac(data_to_seal, key=hmac_key_bytes) 

        if new_mac != row['Integrity_MAC']:
            print(f"! TAMPERING DETECTED @ Activity ID/Row {index + 1}!")
            print(f"   Original MAC: {row['Integrity_MAC'][:15]}...")
            print(f"   Calculated MAC: {new_mac[:15]}...")
            tamper_detected += 1

    if tamper_detected == 0:
        print("✅ No Tampering Detected !!\n")
    else:
        print(f"❌ TAMPERED: {tamper_detected} rows show signs of tampering.\n")


# VERIFICATION AND USER INPUT -----------------------------

if __name__ == "__main__":
    
    print("\nVerifying HMAC.....")
    key_input = input("Please enter the HMAC Key:\n\t>")
    
    try:
        hmac_key_bytes = eval(key_input)
        if not isinstance(hmac_key_bytes, bytes) or len(hmac_key_bytes) != 32:
             raise ValueError("Key must be a 32-byte Python bytes")
        
    except (SyntaxError, ValueError, NameError) as e:
        print(f"\nERROR: Invalid Format {e}")
        print("Make sure you are including the 'b'' prefix.")
        exit()

    print("\nIntegrity Verification Ongoing...")
    verify_data_integrity("Security_Artifacts/ML_encrypted_results.csv", hmac_key_bytes)