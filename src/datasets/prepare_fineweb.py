import argparse

from transformers import AutoTokenizer

from config import DataConfig
from src.datasets.preprocessing_fineweb import tokenize_and_save_dataset


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceFW/ablation-model-fineweb-edu")
    tokenize_and_save_dataset(
        dataset_name="HuggingFaceTB/smollm-corpus",
        split="train",
        subset="fineweb-edu-dedup",
        tokenizer=tokenizer,
        seq_len=2048,
        text_field_name="text",
        num_rows_to_sample=50_000_000,
        cache_dir=args.data_path,
        batch_size=32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DataConfig().data_path, help='Path to the dataset')
    args = parser.parse_args()
    main(args)