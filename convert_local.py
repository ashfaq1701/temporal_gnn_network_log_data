import concurrent.futures
import pandas as pd
from dotenv import load_dotenv


def process_file(i):
    filepath = f"/Volumes/G-DRIVE/traces/preprocessed/CallGraph_{i}.parquet"
    df = pd.read_parquet(filepath)
    selected_df = df[['timestamp', 'traceid', 'service', 'um', 'dm', 'rt']]
    selected_df.to_parquet(f"/Volumes/G-DRIVE/traces/preprocessed_short/CallGraph_{i}.parquet")
    print(f"Processed {filepath}")


if __name__ == "__main__":
    load_dotenv()

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(process_file, i) for i in range(6720)]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")
