# -*- coding: utf-8 -*-
"""
 ASR Training with Wav2Vec2-XLS-R-300M on Synthetic Data
FIXED VERSION for dataset format issues
"""
import numpy as np
import json
import re
import unicodedata
import torch
from dataclasses import dataclass
from typing import Dict, List, Union

from datasets import load_from_disk, Audio
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
batch_size = 4
FREEZE_FEATURE_ENCODER = True  # Set to False to unfreeze

# ==================== LOAD DATASET ====================
print("Loading dataset...")
dataset = load_from_disk(dataset_path)

# CRITICAL FIX: Cast audio column to ensure proper loading
print("Ensuring audio is properly formatted...")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

train_data = dataset["train"]
test_data = dataset["test"]
print(f"? Train: {len(train_data)}, Test: {len(test_data)}")

# Test audio loading
print("\nTesting audio loading...")
try:
    sample = train_data[0]
    audio_test = sample["audio"]["array"]
    print(f"? Audio loads correctly: shape={len(audio_test)}, max={np.max(np.abs(audio_test)):.4f}")
except Exception as e:
    print(f"? ERROR: Cannot load audio! {e}")
    exit(1)

# ==================== CLEAN TEXT ====================
def remove_special_characters(batch):
    """Clean text - ONLY keep Devanagari + space"""
    text = batch["sentence"]
    
    # Handle empty/None
    if not text or len(str(text).strip()) == 0:
        batch["sentence"] = " "
        return batch
    
    # Normalize
    text = unicodedata.normalize("NFC", str(text))
    
    # Devanagari only
    devanagari_range = range(0x0900, 0x097F)
    cleaned = "".join([
        c for c in text 
        if c == " " or ord(c) in devanagari_range
    ])
    
    # Clean whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # If empty after cleaning, use space
    batch["sentence"] = (cleaned + " ") if cleaned else " "
    return batch

print("\nCleaning text (removing numbers, punctuation, English)...")
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

vocab_list = sorted(list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0])))

print(f"Total characters: {len(vocab_list)}")

# Validate
devanagari_range = range(0x0900, 0x097F)
non_devanagari = [c for c in vocab_list if c != " " and ord(c) not in devanagari_range]
if non_devanagari:
    print(f"??  Found {len(non_devanagari)} non-Devanagari chars (shouldn't happen after cleaning!):")
    print(f"   {non_devanagari[:10]}")

# Build vocab dict
vocab_dict = {v: k for k, v in enumerate(vocab_list)}

# Replace space with |
if " " in vocab_dict:
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

# Add special tokens
#vocab_dict["[UNK]"] = len(vocab_dict)
#vocab_dict["[PAD]"] = len(vocab_dict)

print(f"Final vocab size: {len(vocab_dict)} (Devanagari: {len(vocab_dict)-3})")

# Save
with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

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
    
    # Process audio
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    # Process text
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    
    return batch

print("Preparing datasets...")
train_data = train_data.map(prepare_dataset, remove_columns=train_data.column_names)
test_data = test_data.map(prepare_dataset, remove_columns=test_data.column_names)

# ==================== FILTER ====================
def is_valid_sample(sample):
    has_labels = len(sample["labels"]) > 2
    has_audio = sample["input_length"] > 16000  # At least 1s
    
    if has_labels and has_audio:
        audio_sec = sample["input_length"] / 16000
        text_chars = len(sample["labels"])
        ratio = audio_sec / text_chars if text_chars > 0 else 999
        has_good_ratio = 0.02 < ratio < 0.8  # Wide range for synthetic
    else:
        has_good_ratio = False
    
    return has_labels and has_audio and has_good_ratio

print("\nFiltering invalid samples...")
orig_train = len(train_data)
orig_test = len(test_data)

train_data = train_data.filter(is_valid_sample)
test_data = test_data.filter(is_valid_sample)

print(f"Removed: Train {orig_train - len(train_data)}, Test {orig_test - len(test_data)}")
print(f"? Final: Train {len(train_data)}, Test {len(test_data)}\n")

if len(train_data) < 100:
    print("??  WARNING: Very few training samples!")

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

# ==================== METRICS ====================
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    # Debug: Check logits
    print(f"\nDEBUG INFO:")
    print(f"  Logit shape: {pred_logits.shape}")
    print(f"  Logit range: [{np.min(pred_logits):.2f}, {np.max(pred_logits):.2f}]")
    print(f"  Max predicted IDs: {np.max(pred_ids)}, Vocab size: {len(processor.tokenizer)}")
    print(f"  Unique pred IDs: {len(np.unique(pred_ids))}")
    
    # Check if model is only predicting blank/padding
    blank_token = processor.tokenizer.pad_token_id
    blank_ratio = np.mean(pred_ids == blank_token)
    print(f"  Blank token ({blank_token}) ratio: {blank_ratio:.2%}")
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # Check for empty predictions
    empty_preds = sum(1 for p in pred_str if len(p.strip()) == 0)
    if empty_preds > len(pred_str) * 0.5:
        print(f"  ??  WARNING: {empty_preds}/{len(pred_str)} predictions are EMPTY!")
        print(f"  First 5 pred_ids sequences: {pred_ids[0][:20]}")
    
    wer = calculate_wer(label_str, pred_str)
    
    print(f"\n{'='*70}")
    print(f"Prediction: {pred_str[0][:80] if pred_str[0] else '(EMPTY!)'}...")
    print(f"Reference:  {label_str[0][:80]}...")
    print(f"WER: {wer:.4f} ({wer*100:.1f}%)")
    print(f"{'='*70}\n")
    
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
    print("? Feature encoder will be fine-tuned (RECOMMENDED for synthetic data)")

# ==================== TEST MODEL OUTPUT ====================
print("\n" + "="*70)
print("PRE-TRAINING MODEL TEST")
print("="*70)

model.eval()
sample = test_data[0]
test_audio = torch.tensor(sample["input_values"]).unsqueeze(0)

with torch.no_grad():
    outputs = model(test_audio)
    logits = outputs.logits
    
    print(f"Logit shape: {logits.shape}")
    print(f"Logit range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Tokenizer vocab size: {len(processor.tokenizer)}")
    
    if model.config.vocab_size != len(processor.tokenizer):
        print(f"\n? VOCAB SIZE MISMATCH!")
        print(f"   This WILL cause empty predictions!")
        exit(1)
    
    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    pred_text = processor.batch_decode([pred_ids])[0]
    
    print(f"Sample prediction: '{pred_text[:50]}'")
    
    if len(pred_text.strip()) == 0:
        print(f"??  WARNING: Empty prediction before training!")
        print(f"   This might improve during training...")
    else:
        print(f"? Model produces output")

model.train()
print("="*70 + "\n")

# ==================== TRAINING ====================
training_args = TrainingArguments(
    output_dir="./xlsr_300m__synthetic",
    group_by_length=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4,
    
    eval_strategy="epoch",
    save_strategy="epoch",
    
    num_train_epochs=20,
    
    gradient_checkpointing=True,
    fp16=True,
    
    logging_steps=100,
    learning_rate=5e-5,  # Slightly higher since unfrozen
    warmup_steps=1000,  # More warmup
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    
    # Important: Report NaN/Inf
    logging_nan_inf_filter=False,
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
print(f"Dataset: Synthetic ")
print(f"Samples: {len(train_data)} train, {len(test_data)} test")
print(f"Feature encoder: {'FROZEN' if FREEZE_FEATURE_ENCODER else 'UNFROZEN'}")
print(f"{'='*70}\n")

trainer.train()

print(f"\n{'='*70}")
print("TRAINING COMPLETE!")
print(f"{'='*70}\n")