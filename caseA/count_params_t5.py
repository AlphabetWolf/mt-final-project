from transformers import AutoModelForSeq2SeqLM
import adapters

MODEL_NAME = "t5-small"
ADAPTER_NAME = "wmt14_en_fr_adapter"

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    full_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    total_full, train_full = count_params(full_model)
    print("Full FT T5 (baseline) - total:", total_full, "trainable:", train_full)

    adapter_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    adapters.init(adapter_model)

    if ADAPTER_NAME not in adapter_model.adapters_config:
        adapter_model.add_adapter(ADAPTER_NAME, config="pfeiffer")

    adapter_model.train_adapter(ADAPTER_NAME)

    total_adp, train_adp = count_params(adapter_model)
    print("Adapter T5 - total:", total_adp, "trainable:", train_adp)

    print("Trainable ratio (adapters / full):", train_adp / train_full)

if __name__ == "__main__":
    main()
