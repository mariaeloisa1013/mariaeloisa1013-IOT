import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import os 

file_paths = ['Personal_Data/anon1.csv', 'Personal_Data/anon2.csv', 'Personal_Data/anon3.csv']
dataframes = [pd.read_csv(file, encoding='latin-1', skipinitialspace=True) 
              for file in file_paths]

# MERGE DATASETS ----------------------------------
df_master = pd.concat(dataframes, ignore_index=True) 

# removing duplicate columns
df_master = df_master.loc[:,~df_master.columns.duplicated(keep='first')].copy() 
print(f"\n! Merged Datasets: SHAPE {df_master.shape}")

# renaming all ID columns to 'Activity ID'
df_master = df_master.rename(columns={'id': 'Activity ID'}, errors='ignore') 


cols_to_drop = [ # drop unnecessary columns
    'Activity Date', 'Activity Name', 'Activity Description', 'Relative Effort', 'Commute',
    'Activity Private Note', 'Activity Gear', 'Filename', 'Athlete Weight', 'Bike Weight',
    'Elapsed Time', 
    'Distance',     
    'Max Temperature', 'Average Temperature', 'Relative Effort.1', 'Total Work', 
    'Number of Runs', 'Uphill Time', 'Downhill Time', 'Other Time', 'Perceived Exertion',
    'Type', 'Start Time', 'Weighted Average Power', 'Power Count', 'Prefer Perceived Exertion',
    'Perceived Relative Effort', 'Total Weight Lifted', 'From Upload', 'Grade Adjusted Distance',
    'Weather Observation Time', 'Weather Condition', 'Weather Temperature', 'Apparent Temperature',
    'Dewpoint', 'Humidity', 'Weather Pressure', 'Wind Speed', 'Wind Gust', 'Wind Bearing',
    'Precipitation Intensity', 'Sunrise Time', 'Sunset Time', 'Moon Phase', 'Bike', 'Gear',
    'Precipitation Probability', 'Precipitation Type', 'Cloud Cover', 'Weather Visibility',
    'UV Index', 'Weather Ozone', 'Jump Count', 'Total Grit', 'Average Flow', 'Flagged',
    'Dirt Distance', 'Newly Explored Distance', 'Newly Explored Dirt Distance',
    'Activity Count', 'Total Steps', 'Carbon Saved', 'Pool Length', 'Training Load',
    'Intensity', 'Average Grade Adjusted Pace', 'Timer Time', 'Total Cycles', 'Recovery',
    'With Pet', 'Competition', 'Long Run', 'For a Cause', 'Media',

]

df_clean = df_master.drop(columns=cols_to_drop, errors='ignore')


df_clean = df_clean.rename(columns={'Elapsed Time.1': 'Elapsed Time',
                                    'Distance.1': 'Distance'}, errors='ignore')
print(f"\n! Selected Columns: SHAPE {df_clean.shape}")


# CLEAN TIME AND DATES -----------------------------

# convert to datetime objects
df_master['Activity Date'] = pd.to_datetime(df_master['Activity Date'], errors='coerce')

df_clean['Hour_of_Day'] = df_master['Activity Date'].dt.hour
df_clean['Day_of_Week'] = df_master['Activity Date'].dt.day_name()
df_clean['Month'] = df_master['Activity Date'].dt.month

# convert meters to km
df_clean['Distance_km'] = df_clean['Distance'] / 1000
df_clean = df_clean.drop(columns=['Distance'])

print("\n! Successfully cleaned Date and Time Features")


# HANDLE MISSING VALUES -----------------------------

# Define sparse columns for imputation AND flagging
sparse_sensor_cols = ['Max Heart Rate', 'Max Cadence', 'Average Cadence',
                      'Max Watts', 'Average Watts', 'Calories', 'Average Heart Rate']

# Binary Flag Feature Engineering
for col in sparse_sensor_cols:
    df_clean[f'{col}_Recorded'] = df_clean[col].notna().astype(int)
print("! Created Binary Flag features for sparse sensor data.")

# impute 0
for col in sparse_sensor_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
# median IMPUTATION
numerical_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
for col in numerical_cols:
    # median imputation
    df_clean[col] = df_clean[col].fillna(df_clean[col].median()) 

print("! Imputed NAN values")


# PREPARE FEATURES AND TARGET -----------------------------

df_clean = df_clean.dropna(subset=['Activity Type', 'Activity ID']) 

# label encoder to convert target variable to integers
le = LabelEncoder()
Y = le.fit_transform(df_clean['Activity Type'])

# separate X and Y target
X = df_clean.drop(columns=['Activity Type'])

# zero variance filter
X = X.loc[:, X.nunique() > 1]
Activity_Type = df_clean['Activity Type'].reset_index(drop=True)
Activity_ID = df_clean['Activity ID'].reset_index(drop=True) 

print(f"\nTarget Classes Converted: {le.classes_}")
print(f"Shape after Preparing Features and Target: {X.shape}")


# ONE-HOT ENCODING AND SCALING -----------------------------

numerical_features = X.select_dtypes(include=np.number).columns.tolist()

if 'Activity ID' in numerical_features:
    numerical_features.remove('Activity ID')
    
categorical_features = ['Day_of_Week'] 
if 'Day_of_Week' not in X.columns:
    categorical_features = [] 

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop' 
)

X_processed = preprocessor.fit_transform(X)


num_names = [name for name in numerical_features]
cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
feature_names = num_names + cat_names


# CREATE FINAL CSV ---------------------------------

# convert NumPy back to DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# include Activity ID and Type back
X_processed_df.insert(loc=0, column='Activity ID', value=Activity_ID.values)
X_processed_df.insert(loc=1, column='Activity Type', value=Activity_Type.values)


final_csv_filename = "FINALpersonaldataset.csv"
X_processed_df.to_csv(final_csv_filename, index=False)

# LABEL ENCODER
joblib.dump(le, "LabelEncoder.pkl")
# PREPROCESSOR
joblib.dump(preprocessor, "PreProcessor.pkl") 

print(f"\nFINAL SHAPE: {X_processed_df.shape[1]} features")
print(f"\tFinal Dataset saved at: {final_csv_filename}\n")