# region: Data Ingestion
# Downloads the Bitext customer-support dataset from Hugging Face (once),
# and saves it locally as a CSV so future runs don't re-download it.

from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_FILE = DATA_PATH / "bitext_customer_support_raw.csv"


def download_dataset() -> None:
    if RAW_FILE.exists():
        print(f"Dataset already exists at {RAW_FILE}, skipping download.")
        return

    print("Downloading Bitext customer-support dataset from Hugging Face...")
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    df = dataset.to_pandas()
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_FILE, index=False)
    print(f"Saved {len(df)} rows to {RAW_FILE}")


if __name__ == "__main__":
    download_dataset()
# endregion