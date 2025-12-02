# Low-Resource MT with Parameter-Efficient Fine-Tuning

This project explores how **parameter-efficient fine-tuning (PEFT)** compares to **full fine-tuning** for low-resource neural machine translation and domain adaptation.

- **Case A – WMT14 En→Fr (T5-small)**  
  We fine-tune T5-small on a 50k-pair subset of WMT14 English→French.  
  We compare:
  - Full fine-tuning of all model parameters.
  - Encoder–decoder adapters that update only ~0.66% of parameters.  

  Adapters achieve dev BLEU close to full fine-tuning while using far fewer trainable parameters, showing that PEFT is an efficient alternative when compute or memory is limited.

- **Case B – Bible → Medical En→De (Helsinki-NLP/opus-mt-en-de)**  
  Starting from a Bible-domain En→De model, we adapt to a tiny medical-domain parallel corpus.  
  We compare:
  - Full fine-tuning on 2k medical sentence pairs.  
  - LoRA adaptation on the same data.  
  - LoRA with additional synthetic medical pairs generated via back-translation.  

  Full fine-tuning overfits and yields low BLEU, while LoRA and especially LoRA + back-translation achieve much higher BLEU with far fewer updated parameters. This highlights PEFT + synthetic data as a practical recipe for low-resource domain adaptation.

The repository contains scripts to prepare datasets, train the models, evaluate BLEU, and generate plots for training dynamics and parameter efficiency.
