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

MODEL_NAME = "t5-small"
DATA_DIR = "data/wmt14_en_fr_t5_tok"
OUTPUT_DIR = "runs/adapter_en_fr_t5"

class AdapterSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        cleaned_inputs = dict(inputs)
        cleaned_inputs.pop("num_items_in_batch", None)
        return super().compute_loss(model, cleaned_inputs, return_outputs=return_outputs)


def main():
    print("Loading tokenized dataset...")
    dataset = load_from_disk(DATA_DIR)
    print(dataset)

    print("Loading tokenizer and base model (T5)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    adapters.init(model)

    adapter_name = "wmt14_en_fr_adapter"

    if adapter_name not in model.adapters_config:
        model.add_adapter(adapter_name, config="pfeiffer")

    model.train_adapter(adapter_name)
    model.set_active_adapters(adapter_name)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    fp16 = torch.cuda.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        fp16=fp16,
    )

    print("Starting adapter fine-tuning (T5 + adapters)...")
    trainer = AdapterSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    print("Training finished, saving adapter model and tokenizer...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved adapter T5 model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
