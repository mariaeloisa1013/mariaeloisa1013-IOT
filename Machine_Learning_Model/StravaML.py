import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Import Security Library
from SecurityLibrary import generate_hmac, encrypt_id, get_fernet_key, get_hmac_key 

# Load datasets
df_train = pd.read_csv("../Data_Preprocessing/FINALpublicdataset.csv") # For Training
df_predict = pd.read_csv("../Data_Preprocessing/FINALpersonaldataset.csv") # For Testing
features = [
    "Distance_km", 
    "Elapsed Time", 
    "Moving Time", 
    "Average Speed", 
    "Elevation Gain", 
    "Max Speed",
    "Max Grade",
    "Average Grade"
]
print(f"Columns available in df_predict: {df_predict.columns.tolist()}")

# MODEL -------------------------------------

# Splitting the data for training 
X_train = df_train[features] # Independent variables
y_train = df_train["Activity Type"] # Dependent/Target variable

# Splitting the data for testing
X_test = df_predict[features]
y_test = df_predict["Activity Type"] 

# Train the model 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict activity type 
df_predict["Predicted Activity Type"] = model.predict(X_test)

# SECURITY CONFIGURATIONS ------------------------

# Data Masking: Cryptography
df_predict["Encrypted ID"] = df_predict["Activity ID"].astype(str).apply(encrypt_id) 

# Integrity/Authenticity Checker: HMAC (hides the Encrypted ID, the Actual type, and the Predicted type.) 
df_predict['Integrity_MAC'] = df_predict.apply(lambda row: generate_hmac(
    pd.Series([row['Encrypted ID'], row['Activity Type'], row['Predicted Activity Type']])
), axis=1)

# Dataaframe for Verifier
secured_output_filename = "SecuredData.csv"
secured_output_df = df_predict[[
    'Encrypted ID',      
    'Activity Type', 
    'Predicted Activity Type',
    'Integrity_MAC'      
]].copy()

secured_output_df.to_csv(secured_output_filename, index=False)
print(f"\nSaved secured data for audit to {secured_output_filename}")


# RESULTS & SNAPSHOTS -------------------------------------

result = df_predict[features].copy() 

# Insert the security columns and comparison column
result.insert(0, "Encrypted ID", df_predict["Encrypted ID"]) 
result.insert(8, "Actual Activity Type", df_predict["Activity Type"]) 
result["Predicted Activity Type"] = df_predict["Predicted Activity Type"]
result["Integrity_MAC"] = df_predict["Integrity_MAC"]

# Print the enhanced table
print("\n--- Predictions vs. Actual Activity Type (Secured Output) ---\n")
print(result.to_string(index=False))

# Raw Data Snapshot
print("\nSecurity Data Snapshot (First 3 Rows):")
print(df_predict[['Activity ID', 'Encrypted ID', 'Integrity_MAC']].head(3).to_string(index=False))

# EVALUATION METRICS -------------------------------------

# Predict on test set 
y_pred = df_predict["Predicted Activity Type"]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred) 

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report\n")
print(classification_report(y_test, y_pred)) # Show precision, recall, f1-score per class

# Accuracy Chart
accuracy_per_class = cm.diagonal() / cm.sum(axis=1) # Accuracy per class
class_names = model.classes_
overall_accuracy = accuracy_score(y_test, y_pred) # Overall model accuracy
overall_accuracy_pct = overall_accuracy * 100 # Convert to percentage

plt.figure(figsize=(8, 6))
plt.plot(class_names, accuracy_per_class, marker='o', linestyle='-', color='green', linewidth=2, label='Per-Class Accuracy')
plt.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall Accuracy ({overall_accuracy_pct:.2f}%)')

# Add value labels above points
for i, v in enumerate(accuracy_per_class):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
plt.title("Accuracy Chart")
plt.xlabel("Activity Type")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\nCompleted all tasks successfully.")
print(f"Overall Model Accuracy: {overall_accuracy_pct:.2f}%\n") # Print overall accuracy

# SECURITY KEYS -------------------------------------

print(f"\nFERNET ENCRYPTION KEY (for decryption): {get_fernet_key().decode()}")
print(f"\nHMAC SECRET KEY (for verification): {get_hmac_key()}\n")