import pandas as pd
import os

# For Files with Inconsistent Date Format
INPUT_FILENAME = 'Personal_Data/anon4.csv'
OUTPUT_FILENAME = 'Personal_Data/anon4.csv'
INPUT_DATE_FORMAT = "%b %d, %Y"

# transformed format: Month Day, Year 
OUTPUT_DATE_FORMAT = "%b %d, %Y"


def process_activity_dates(input_file, output_file):
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please ensure CSV file path")
        return

    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        df['Activity Date'] = pd.to_datetime(df['Activity Date'], format=INPUT_DATE_FORMAT)
        df['Activity Date'] = df['Activity Date'].dt.strftime(OUTPUT_DATE_FORMAT)
        df.to_csv(output_file, index=False)
        
        print("\n !! DONE ")
        print(f"\tDate format changed for {len(df)} activities.")
        print(f"\tNew data saved to: {output_file}")
        print(f"\tExample new date format: {df['Activity Date'].iloc[0]}\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    process_activity_dates(INPUT_FILENAME, OUTPUT_FILENAME)
