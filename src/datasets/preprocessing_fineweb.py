from datasets import Dataset, load_dataset, load_from_disk
import multiprocessing
import numpy as np


def tokenize_and_save_dataset(dataset_name: str, split: str, subset: str, tokenizer, seq_len: int, text_field_name: str, num_rows_to_sample: int, cache_dir: str, batch_size: int) -> Dataset:
    num_proc = multiprocessing.cpu_count()
    print('Batch size for preprocessing:', batch_size)

    dataset = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir, num_proc=num_proc).select_columns(text_field_name)
    print("Full dataset size:", len(dataset))

    dataset = dataset.select(range(num_rows_to_sample))
    print("Subset size before processing:", len(dataset))

    print("Tokenizing data...")
    dataset = dataset.map(lambda x: tokenizer(x[text_field_name], truncation=False, padding=False, verbose=False), batched=True, num_proc=num_proc, remove_columns=text_field_name, batch_size=batch_size)

    def batched_split(batch):
        all_input_ids = []
        for sentence in batch["input_ids"]:
            all_input_ids += [tokenizer.bos_token_id] + sentence + [tokenizer.eos_token_id]

        # dropping the remainder
        total_length = (len(all_input_ids) // seq_len) * seq_len
        if total_length == len(all_input_ids):
            total_length -= seq_len

        # splitting to equal parts
        input_ids = [all_input_ids[i: i + seq_len] for i in range(0, total_length, seq_len)]
        labels = [all_input_ids[i + 1: i + seq_len + 1] for i in range(0, total_length, seq_len)]

        return {"input_ids": input_ids, "labels": labels}

    print("Splitting sequences...")
    dataset = dataset.map(batched_split, batched=True, num_proc=num_proc, remove_columns=["attention_mask"], batch_size=batch_size)
    dataset = dataset.shuffle(seed=42)
    dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    num_tokens = len(dataset) * seq_len
    print(f"num tokens {num_tokens}")

    print("Saving dataset...")
    dataset.save_to_disk(f"{cache_dir}/tokenized/{dataset_name.split('/')[-1]}__{subset}/len{seq_len}_{num_tokens}tokens")
    
    print("Dataset length after processing:", len(dataset))
    return dataset