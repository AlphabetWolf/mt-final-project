import os
import json
import glob
import matplotlib.pyplot as plt

FULL_DIR = "runs/full_ft_en_fr_t5"
ADAPTER_DIR = "runs/adapter_en_fr_t5"

# I got them from training scripts' final evals
FULL_DEV_BLEU = 30.70
ADAPTER_DEV_BLEU = 29.96

# I got them from count_params_t5.py
FULL_TRAINABLE_PARAMS = 60_506_624    # full FT
ADAPTER_TRAINABLE_PARAMS = 399_744    # adapters only


def find_trainer_state(run_dir: str) -> str:
    direct_path = os.path.join(run_dir, "trainer_state.json")
    if os.path.exists(direct_path):
        return direct_path

    # Look for trainer_state.json
    pattern = os.path.join(run_dir, "checkpoint-*", "trainer_state.json")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"No trainer_state.json found in {run_dir} "
            f"(looked at {direct_path} and {pattern})"
        )

    def extract_step(path):
        ckpt_dir = os.path.basename(os.path.dirname(path))
        try:
            return int(ckpt_dir.split("-")[-1])
        except ValueError:
            return -1

    candidates.sort(key=extract_step)
    return candidates[-1]


def load_run(run_dir, label):
    trainer_state_path = find_trainer_state(run_dir)
    print(f"[{label}] Using trainer_state.json from: {trainer_state_path}")

    with open(trainer_state_path, "r") as f:
        state = json.load(f)

    logs = state.get("log_history", [])

    train_steps, train_loss = [], []
    eval_epochs, eval_bleu = [], []

    for entry in logs:
        if "loss" in entry and "step" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])

        if "eval_bleu" in entry:
            eval_epochs.append(entry.get("epoch", None))
            eval_bleu.append(entry["eval_bleu"])

    return {
        "label": label,
        "train_steps": train_steps,
        "train_loss": train_loss,
        "eval_epochs": eval_epochs,
        "eval_bleu": eval_bleu,
    }


def plot_training_loss(full_run, adapter_run, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(full_run["train_steps"], full_run["train_loss"],
             label="Full FT (all params)")
    plt.plot(adapter_run["train_steps"], adapter_run["train_loss"],
             label="Adapters (0.66% params)")

    plt.xlabel("Training step")
    plt.ylabel("Training loss (cross-entropy)")
    plt.title("T5-small on WMT14 En-Fr (50k): training loss vs. step")
    plt.legend(title="Fine-tuning method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_bleu_and_params(out_bleu_path, out_params_path):
    methods = ["Full FT (all params)", "Adapters (0.66% params)"]
    bleu_values = [FULL_DEV_BLEU, ADAPTER_DEV_BLEU]

    plt.figure(figsize=(5, 4))
    plt.bar(methods, bleu_values)
    plt.ylabel("Validation BLEU (sacreBLEU)")
    plt.title("T5-small on WMT14 En-Fr: dev BLEU")

    plt.ylim(28.5, 31.5)
    for i, v in enumerate(bleu_values):
        plt.text(i, v + 0.05, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_bleu_path, dpi=200)
    plt.close()

    params_m = [
        FULL_TRAINABLE_PARAMS / 1e6,
        ADAPTER_TRAINABLE_PARAMS / 1e6,
    ]

    plt.figure(figsize=(5, 5))
    plt.bar(methods, params_m)
    plt.ylabel("Trainable parameters (millions)")
    plt.title("T5-small on WMT14 En-Fr: trainable parameters")

    for i, v in enumerate(params_m):
        plt.text(i, v + 1, f"{v:.2f}M",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_params_path, dpi=200)
    plt.close()

def main():
    os.makedirs("figures", exist_ok=True)

    full_run = load_run(FULL_DIR, "Full FT")
    adapter_run = load_run(ADAPTER_DIR, "Adapters")

    plot_training_loss(
        full_run,
        adapter_run,
        "figures/t5_wmt14_train_loss.png",
    )

    plot_bleu_and_params(
        "figures/t5_wmt14_bleu_bar.png",
        "figures/t5_wmt14_trainable_params_bar.png",
    )

    print("Saved plots in ./figures/")


if __name__ == "__main__":
    main()
