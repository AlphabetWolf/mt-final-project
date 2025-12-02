import numpy as np
import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

MODEL_NAME = "t5-small"
DATA_DIR = "data/wmt14_en_fr_t5_tok"
OUTPUT_DIR = "runs/full_ft_en_fr_t5"

def main():
    print("Loading tokenized dataset...")
    dataset = load_from_disk(DATA_DIR)
    print(dataset)

    print("Loading tokenizer and base model (T5)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    bleu = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [[l.strip()] for l in labels]
        return preds, labels

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    fp16 = torch.cuda.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        fp16=fp16,
    )

    print("Starting full fine-tuning (T5)...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # print("Final evaluation on validation set:")
    # metrics = trainer.evaluate()
    # print(metrics)

    # Save final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved full FT T5 model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
