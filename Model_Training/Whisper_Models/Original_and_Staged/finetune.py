from datasets import load_from_disk
from transformers import (
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from jiwer import wer as calculate_wer
import os

# Optional: Set specific GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ==================== CONFIG ====================
data_path = "path"
model_path = "path"

# ==================== LOAD DATASET ====================
print("Loading dataset...")
dataset = load_from_disk(data_path)
print(f"? Train samples: {len(dataset['train'])}")
print(f"? Test samples: {len(dataset['test'])}")

# ==================== LOAD MODEL & PROCESSOR ====================
print("\nLoading model and processor...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(
    model_path, 
    language="Hindi",  # Use Hindi tokenizer for Devanagari script
    task="transcribe"
)
processor = WhisperProcessor.from_pretrained(
    model_path, 
    language="Hindi", 
    task="transcribe"
)

print("? Processor loaded")

# ==================== PREPARE DATASET ====================
def prepare_dataset(batch):
    """Prepare audio and text for Whisper"""
    audio = batch["audio"]

    # Compute log-Mel input features
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    
    return batch

print("\nPreparing datasets...")
dataset = dataset.map(prepare_dataset, remove_columns=["audio", "sentence"])
print("? Datasets prepared")

train_data = dataset["train"]

x = len(train_data)
wndw = x // 5

train_data = train_data.select(range(4*wndw))
test_data = dataset["test"]


# ==================== LOAD MODEL ====================
print("\nLoading Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained(model_path)

# Disable cache for training
model.config.use_cache = False
model.generation_config.use_cache = False

# Set generation config
model.generation_config.language = "hindi"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

print("? Model loaded")

# ==================== DATA COLLATOR ====================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, 
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split inputs and labels (different lengths, different padding)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), 
            -100
        )

        # If bos token is appended, remove it (will be added by model)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# ==================== METRICS (JIWER WER) ====================
def compute_metrics(pred):
    """
    Compute WER using jiwer library
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    # skip_special_tokens=True removes <|startoftranscript|>, <|endoftext|>, etc.
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate WER using jiwer
    wer = calculate_wer(label_str, pred_str)
    
    # Print sample for monitoring
    print(f"\n{'='*70}")
    print(f"Sample Prediction: {pred_str[0][:100]}...")
    print(f"Sample Reference:  {label_str[0][:100]}...")
    print(f"WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"{'='*70}\n")

    # Return as percentage for better readability
    return {"wer": wer * 100}  # Convert to percentage

# ==================== TRAINING ARGUMENTS ====================
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper_medium__normalized_30_epochs",
    
    # Batch sizes
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,  # Set to 2 as requested
    gradient_accumulation_steps=1,
    
    # Training schedule
    num_train_epochs=30,
    learning_rate=1e-5,
    warmup_steps=500,  # Recommended for Whisper
    
    # Evaluation & Saving
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    
    # Generation settings for evaluation
    predict_with_generate=True,  # IMPORTANT: Enable for WER calculation
    generation_max_length=448,
    
    # Performance
    gradient_checkpointing=False,  # Enable to save memory
    fp16=True,
    
    # Model selection
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,  # Lower WER is better
    
    # Saving
    save_total_limit=3,  # Keep only best 3 checkpoints
    
    # Logging
    report_to=["tensorboard"],
    push_to_hub=False,
)

# ==================== TRAINER ====================
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # Enable WER evaluation
    tokenizer=processor.feature_extractor,
)

# Save processor for inference later
processor.save_pretrained(training_args.output_dir)

print(f"\n{'='*70}")
print("STARTING WHISPER FINE-TUNING")
print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Eval batch size: 2")
print(f"{'='*70}\n")

# ==================== TRAIN ====================
trainer.train()

print(f"\n{'='*70}")
print("TRAINING COMPLETE!")
print(f"Best model saved to: {training_args.output_dir}")
print(f"{'='*70}\n")

# ==================== FINAL EVALUATION ====================
print("\nRunning final evaluation...")
metrics = trainer.evaluate()

print(f"\n{'='*70}")
print("FINAL EVALUATION METRICS")
print(f"{'='*70}")
for key, value in metrics.items():
    print(f"{key}: {value}")
print(f"{'='*70}\n")