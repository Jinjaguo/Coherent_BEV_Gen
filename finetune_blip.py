import torch, json
from datasets import load_dataset, Features, Value, Image
from transformers import (InstructBlipProcessor,
                          InstructBlipForConditionalGeneration,
                          TrainingArguments, Trainer,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

# ---------- 1. 数据 ----------
features = Features({
    "image": Image(),
    "prompt": Value("string"),
    "answer": Value("string"),
})
ds = load_dataset("json", data_files="train.jsonl", features=features)["train"]

# ---------- 2. 模型 ----------
base_dir = "./InstructBLIP"
processor = InstructBlipProcessor.from_pretrained(base_dir, use_fast=False)
model = InstructBlipForConditionalGeneration.from_pretrained(
            base_dir, torch_dtype=torch.float16, device_map="auto")

# ---------- 3. LoRA ----------
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["query","key","value","dense",   # Q‑Former
                    "q","k","v","o"],                # T5 Self/Cross‑Attn
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ---------- 4. collate_fn：一次性批量编码 ----------
def collate_fn(batch):
    imgs     = [b["image"]  for b in batch]
    prompts  = [b["prompt"] for b in batch]
    answers  = [b["answer"] for b in batch]

    # ① 编码 图像 + prompt（含 qformer_*、pixel_values 等）
    model_inputs = processor(
        images=imgs,
        text=prompts,
        padding=True,
        return_tensors="pt"
    )

    # ② 编码 answer 作为 labels
    labels = processor.tokenizer(
        answers,
        padding=True,
        return_tensors="pt"
    ).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

# ---------- 5. 训练参数 ----------
args = TrainingArguments(
    output_dir="./lora_ckpt",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=8,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    remove_unused_columns=False,
)

# ---------- 6. Trainer ----------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collate_fn,

)
trainer.train()

# ---------- 7. 保存 ----------
model.save_pretrained("./InstructBLIP/lora_adapter")
processor.save_pretrained("./InstructBLIP/lora_adapter")
print("✅ LoRA 适配器已保存到 ./InstructBLIP/lora_adapter")