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
import adapters
import evaluate

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
DATA_DIR = "data/wmt14_en_fr_tok"
OUTPUT_DIR = "runs/adapter_en_fr"

def main():
    print("Loading tokenized dataset...")
    dataset = load_from_disk(DATA_DIR)
    print(dataset)

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    adapters.init(model)

    adapter_name = "wmt14_domain"

    if adapter_name not in model.adapters_config:
        model.add_adapter(adapter_name, config="seq_bn")

    model.train_adapter(adapter_name)
    model.set_active_adapters(adapter_name)

    bleu = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [[l.strip()] for l in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
        result["gen_len"] = float(np.mean(prediction_lens))
        return {k: round(v, 4) for k, v in result.items()}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    fp16 = torch.cuda.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        save_total_limit=2,
        fp16=fp16,
        report_to="none",
    )

    print("Starting adapter fine-tuning (only adapter params trainable)...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Final evaluation on validation set:")
    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved adapter model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
