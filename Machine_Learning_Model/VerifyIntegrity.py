import pandas as pd
import hmac
import hashlib
import os


# The key is now passed as an argument
def generate_hmac(data_series: pd.Series, key: bytes) -> str:
    """Re-generates the HMAC for verification."""
    data_str = data_series.astype(str).str.cat(sep='|')
    data_bytes = data_str.encode('utf-8')
    
    # Use the key provided to the function
    hm = hmac.new(key, data_bytes, hashlib.sha256)
    return hm.hexdigest()

def verify_data_integrity(file_path, hmac_key_bytes):
    """Loads the data and verifies the Integrity_MAC for each row."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{file_path}'.")
        return

    tamper_detected = 0
    
    for index, row in df.iterrows():
        # 1. Define the data columns used in the original sealing process
        data_to_seal = pd.Series([
            row['Encrypted ID'], 
            row['Activity Type'], 
            row['Predicted Activity Type']
        ])
        
        # 2. Re-calculate the seal using the ORIGINAL key provided at runtime
        # Pass the key to the generate_hmac function
        new_mac = generate_hmac(data_to_seal, key=hmac_key_bytes) 
        
        # 3. Compare the new seal against the saved seal
        if new_mac != row['Integrity_MAC']:
            # The row index + 1 gives the Activity ID/Row number
            print(f"🚨 TAMPERING DETECTED on Activity ID/Row {index + 1}!")
            print(f"   Original MAC: {row['Integrity_MAC'][:15]}...")
            print(f"   Calculated MAC: {new_mac[:15]}...")
            tamper_detected += 1

    if tamper_detected == 0:
        print("\n✅ Integrity Check PASSED! No tampering detected on the data.")
    else:
        print(f"\n❌ CHECK FAILED: {tamper_detected} rows show signs of tampering.")


# --- Run Verification and Key Input ---
if __name__ == "__main__":
    
    print("\n--- HMAC Verification Setup ---")
    
    # 1. Ask the user to input the secret key
    key_input = input("Please enter the HMAC Secret Key (copy the full b'...'):\n> ")
    
    try:
        # 2. Convert the input string back into a Python bytes object
        # We use eval() because the key is a bytes literal (b'...')
        hmac_key_bytes = eval(key_input)
        
        # 3. Check if the key is the correct type and size (32 bytes for HMAC-SHA256)
        if not isinstance(hmac_key_bytes, bytes) or len(hmac_key_bytes) != 32:
             raise ValueError("Key must be a valid 32-byte Python bytes literal (e.g., b'\\x00...').")
        
    except (SyntaxError, ValueError, NameError) as e:
        print(f"\nERROR: Invalid key format provided. {e}")
        print("Please ensure you copy the entire Python bytes literal, including the 'b'' prefix.")
        exit() # Stop execution if the key is invalid

    # 4. Run the verification process with the user-provided key
    print("\nStarting integrity verification...")
    verify_data_integrity("SecuredData.csv", hmac_key_bytes)