import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from safetensors.torch import load_file
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from shared_attention import SharedAttention, SharedMLP, convert_to_recursive
from lora_layer import LoRAAdapter, LoRALinear, LoRAConv1D

# IterableDataset with concatenation into fixed-length chunks
class ChunkedDataset(IterableDataset):
    def __init__(self, hf_stream, tokenizer, block_size=1024):
        self.dataset = hf_stream
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        buffer = []
        buffer_len = 0

        for example in self.dataset:
            ids = self.tokenizer(example["text"]).input_ids
            buffer.extend(ids)
            buffer_len += len(ids)

            # Yield in block_size chunks
            while buffer_len >= self.block_size:
                input_ids = buffer[:self.block_size]
                buffer = buffer[self.block_size:]
                buffer_len = len(buffer)
                yield {"input_ids": torch.tensor(input_ids, dtype=torch.long)}

# DistillationTrainer with KL divergence
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # put inputs on same device as model
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            teacher_logits = teacher_model(input_ids).logits

        student_logits = model(input_ids).logits
        T = 2.0  # temperature

        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        kl_loss = torch.sum(
            teacher_probs * (torch.log(teacher_probs + 1e-8) - student_log_probs),
            dim=-1
        )
        loss = kl_loss.mean()

        return (loss, student_logits) if return_outputs else loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start from base GPT-2
    student_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    global teacher_model
    teacher_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    # Apply recursive modifications
    K = 2
    rank = 8
    student_model = convert_to_recursive(student_model, K=K, rank=rank)

    saved_model_path = "/content/drive/MyDrive/rrt_distilled_streaming/checkpoint-100000/model.safetensors"
    if os.path.exists(saved_model_path):
        print(f"Loading model from {saved_model_path}")
        state_dict = load_file(saved_model_path)
        student_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Model file {saved_model_path} not found. Skipping loading.")

    student_model.to(device)

    print(f"Number of trainable parameters in the student model: {count_parameters(student_model)}")
    print(f"Number of trainable parameters in the teacher model: {count_parameters(teacher_model)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Stream dataset
    raw_stream = load_dataset("Geralt-Targaryen/openwebtext2", split="train", streaming=True)
    train_dataset = ChunkedDataset(raw_stream, tokenizer, block_size=1024)

    # Data collator (no padding needed since already chunked)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./drive/MyDrive/rrt_distilled_streaming",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),  # only enabled if CUDA exists
        logging_steps=100,
        save_steps=5000,
        save_total_limit=2,
        gradient_accumulation_steps=2,
        max_steps=500_000,
        report_to="wandb"
    )

    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

if __name__ == "__main__":
    main()
