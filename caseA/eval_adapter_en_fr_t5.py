import numpy as np
import torch

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

MODEL_DIR = "runs/adapter_en_fr_t5"        # adapter-trained model dir
TOKENIZER_NAME = "t5-small"                # same tokenizer
DATA_DIR = "data/wmt14_en_fr_t5_tok"       # tokenized dataset

def main():
    print("Loading dataset...")
    ds = load_from_disk(DATA_DIR)
    val_ds = ds["validation"]
    print(val_ds)

    print("Loading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    bleu = evaluate.load("sacrebleu")

    all_preds = []
    all_refs = []

    batch_size = 32

    for start in range(0, len(val_ds), batch_size):
        end = min(start + batch_size, len(val_ds))
        batch = val_ds[start:end]

        batch_inputs = tokenizer.pad(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch_inputs["input_ids"].to(device)
        attention_mask = batch_inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
            )

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        refs = []
        for label_seq in batch["labels"]:
            label_seq = [
                (tok if tok != -100 else tokenizer.pad_token_id)
                for tok in label_seq
            ]
            ref_text = tokenizer.decode(label_seq, skip_special_tokens=True)
            refs.append(ref_text)

        all_preds.extend([p.strip() for p in preds])
        all_refs.extend([[r.strip()] for r in refs])

        if (start // batch_size) % 20 == 0:
            print(f"Processed {end} / {len(val_ds)} examples...")

    result = bleu.compute(predictions=all_preds, references=all_refs)
    print(f"Validation BLEU (T5 + adapters): {result['score']:.2f}")


if __name__ == "__main__":
    main()
