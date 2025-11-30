import sys
import pandas as pd
from pathlib import Path

def main(raw_path, processed_path):
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path)
    df = df.dropna()
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

if __name__ == "__main__":
    raw = sys.argv[1]
    processed = sys.argv[2]
    main(raw, processed)
