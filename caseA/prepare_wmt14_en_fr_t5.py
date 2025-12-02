from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "t5-small"
MAX_LEN = 128
OUTPUT_DIR = "data/wmt14_en_fr_t5_tok"

MAX_TRAIN_EXAMPLES = 50000   # set to None for full data
MAX_VAL_EXAMPLES = 3000
MAX_TEST_EXAMPLES = 3000

def main():
    print("Loading raw WMT14 En-Fr dataset...")
    raw = load_dataset("wmt14", "fr-en")  # splits: train, validation, test

    def split_lang(example):
        en = example["translation"]["en"]
        fr = example["translation"]["fr"]
        src = f"translate English to French: {en}"
        return {"src": src, "tgt": fr}

    raw = raw.map(split_lang, remove_columns=raw["train"].column_names)

    if MAX_TRAIN_EXAMPLES is not None:
        raw["train"] = raw["train"].select(range(min(MAX_TRAIN_EXAMPLES, len(raw["train"]))))
    if MAX_VAL_EXAMPLES is not None:
        raw["validation"] = raw["validation"].select(range(min(MAX_VAL_EXAMPLES, len(raw["validation"]))))
    if MAX_TEST_EXAMPLES is not None and "test" in raw:
        raw["test"] = raw["test"].select(range(min(MAX_TEST_EXAMPLES, len(raw["test"]))))

    print(raw)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch["src"],
            max_length=MAX_LEN,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["tgt"],
            max_length=MAX_LEN,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing splits...")
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        num_proc=8,
        remove_columns=["src", "tgt"],
    )

    print(tokenized)

    print(f"Saving tokenized dataset to {OUTPUT_DIR} ...")
    tokenized.save_to_disk(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
