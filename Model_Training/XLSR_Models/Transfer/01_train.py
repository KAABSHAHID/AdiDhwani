# -*- coding: utf-8 -*-
"""
 ASR Training with Wav2Vec2-XLS-R-300M
Fixed version with proper vocab handling
"""
import numpy as np
import json
import re
import unicodedata
import torch
from dataclasses import dataclass
from typing import Dict, List, Union

from datasets import load_from_disk
from jiwer import wer as calculate_wer
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)

# ==================== CONFIG ====================
model_checkpoint = "path"
dataset_path = "path"
batch_size = 8
FREEZE_FEATURE_ENCODER = True  # Set to False to unfreeze

# ==================== LOAD DATASET ====================
print("Loading dataset...")
dataset = load_from_disk(dataset_path)
train_data = dataset["train"]

test_data = dataset["test"]
print(f"? Train: {len(train_data)}, Test: {len(test_data)}")

# ==================== CLEAN TEXT ====================
def remove_special_characters(batch):
    """Clean text - ONLY keep Devanagari + space"""
    text = batch["sentence"]
    
    # Normalize
    text = unicodedata.normalize("NFC", text)
    
    # Devanagari Unicode range
    devanagari_range = range(0x0900, 0x097F)
    
    # Filter: keep only Devanagari + space
    cleaned = "".join([
        c for c in text 
        if c == " " or ord(c) in devanagari_range
    ])
    
    # Clean whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    batch["sentence"] = cleaned + " " if cleaned else " "
    return batch

print("\nCleaning text...")
train_data = train_data.map(remove_special_characters)
test_data = test_data.map(remove_special_characters)
print("? Text cleaned")

# ==================== BUILD VOCAB ====================
def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab]}

print("\nExtracting vocabulary...")
vocab_train = train_data.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=train_data.column_names
)
vocab_test = test_data.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=test_data.column_names
)

# Merge vocabularies
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_list = sorted(vocab_list)

# Validate - should only have Devanagari + space
print(f"\n{'='*60}")
print("VOCABULARY VALIDATION")
print(f"{'='*60}")
devanagari_range = range(0x0900, 0x097F)
for char in vocab_list:
    if char != " " and ord(char) not in devanagari_range:
        print(f"??  WARNING: Non-Devanagari char found: '{char}' (U+{ord(char):04X})")
print(f"Total characters: {len(vocab_list)}")



# Build vocab dict
vocab_dict = {v: k for k, v in enumerate(vocab_list)}

# Replace space with |
if " " in vocab_dict:
    vocab_dict["|"] = vocab_dict.pop(" ")




print(f"\nFinal vocab size: {len(vocab_dict)}")
print(f"  Devanagari chars: {len(vocab_dict) - 3}")


# Check for gaps
indices = sorted([v for v in vocab_dict.values()])
for i in range(len(indices)-1):
    if indices[i+1] - indices[i] != 1:
        print(f"??  GAP DETECTED: Missing index {indices[i]+1}")

# Save
with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
print("? Vocabulary saved to vocab.json\n")

# ==================== PROCESSOR ====================
tokenizer = Wav2Vec2CTCTokenizer(
    "./vocab.json",

    word_delimiter_token="|"
)


feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)

print(f"? Processor created (vocab size: {len(processor.tokenizer)})\n")

# ==================== PREPARE DATASET ====================
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

print("Preparing datasets...")
train_data = train_data.map(prepare_dataset, remove_columns=train_data.column_names)
test_data = test_data.map(prepare_dataset, remove_columns=test_data.column_names)
print("? Datasets prepared\n")

# ==================== DATA COLLATOR ====================
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


wer_save = []
# ==================== METRICS ====================
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = calculate_wer(label_str, pred_str)
    
    print(f"\n{'='*70}")
    print(f"Prediction: {pred_str[0][:100]}...")
    print(f"Reference:  {label_str[0][:100]}...")
    print(f"WER: {wer:.4f}")
    print(f"{'='*70}\n")
    wer_save.append(wer)
    return {"wer": wer}

# ==================== MODEL ====================
print("Loading model...")
model = Wav2Vec2ForCTC.from_pretrained(
    model_checkpoint,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True
)


if FREEZE_FEATURE_ENCODER:
    model.freeze_feature_encoder()
    print("? Feature encoder FROZEN")
else:
    print("? Feature encoder will be fine-tuned")

# ==================== TRAINING ====================
training_args = TrainingArguments(
    output_dir=f"./xlsr_300m__{'frozen' if FREEZE_FEATURE_ENCODER else 'unfrozen'}",
    group_by_length=True,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    
    num_train_epochs=20,
    gradient_checkpointing=True,
    fp16=True,
    #save_steps="epoch",
    #eval_steps="epoch",
    logging_steps=500,
    learning_rate=1e-4,  # Lower LR
    warmup_steps=500,
    max_grad_norm=1.0,   # Gradient clipping
    #save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=processor.feature_extractor,
)


print(f"\n{'='*70}")
print("STARTING TRAINING")
print(f"Feature encoder: {'FROZEN' if FREEZE_FEATURE_ENCODER else 'UNFROZEN'}")
print(f"{'='*70}\n")



trainer.train()

print(f"\n{'='*70}")
print("TRAINING COMPLETE!")
print(f"{'='*70}\n")

print(wer_save)


