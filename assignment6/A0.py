
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Preview CSV structure.")
    parser.add_argument("--csv", default="DCT_mal.csv")
    parser.add_argument("--rows", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print("Columns:", list(df.columns))
    print(df.head(args.rows))

if __name__ == "__main__":
    main()
