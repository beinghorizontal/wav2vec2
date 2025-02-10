import os
import torch
import evaluate
import numpy as np
import random
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, 
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from google.colab import drive, output

# Enable custom widget manager
output.enable_custom_widget_manager()

# Check GPU availability
gpu_info = os.popen('nvidia-smi').read()
if 'failed' in gpu_info:
    print('Not connected to a GPU')
else:
    print(gpu_info)

# Install required packages
os.system('pip install datasets git+https://github.com/huggingface/transformers jiwer librosa evaluate>=0.30 gradio bitsandbytes accelerate')

# Load dataset
import datasets
timit = datasets.load_dataset("crossdelenna/whisper_data_merge2")

# Split dataset
num_rows = len(timit['train'])
num_test_rows = num_rows // 7
num_train_rows = num_rows - num_test_rows
timit_train = timit["train"].select(range(num_train_rows))
timit_test = timit["train"].select(range(num_test_rows))

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Load Whisper components
feature_extractor = WhisperFeatureExtractor.from_pretrained("crossdelenna/medium_cross.en")
tokenizer = WhisperTokenizer.from_pretrained("crossdelenna/medium_cross.en", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("crossdelenna/medium_cross.en", language="English", task="transcribe")

# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Evaluation metric
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Load model
model = WhisperForConditionalGeneration.from_pretrained("crossdelenna/medium_cross.en")

# Freeze layers
def freeze_whisper_layers(model):
    for param in model.parameters():
        param.requires_grad = False

    try:
        encoder_layers = model.model.encoder.layers
        for layer in encoder_layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
    except AttributeError:
        print("Could not access encoder layers")

    try:
        decoder_layers = model.model.decoder.layers
        for layer in decoder_layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
    except AttributeError:
        print("Could not access decoder layers")

    try:
        model.model.encoder.layer_norm.requires_grad = True
    except AttributeError:
        print("Could not access encoder layer norm")

    try:
        model.model.decoder.layer_norm.requires_grad = True
    except AttributeError:
        print("Could not access decoder layer norm")

    for name, module in model.named_children():
        if 'proj' in name or 'head' in name or 'classifier' in name:
            for param in module.parameters():
                param.requires_grad = True

    return model

model = freeze_whisper_layers(model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-medium.en",
    per_device_train_batch_size=22,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=10,
    max_steps=1051,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=22,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=350,
    eval_steps=350,
    logging_steps=350,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id="crossdelenna/medium_cross.en",
    hub_token='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',
)

trainer = Seq2SeqTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=timit_train,
    eval_dataset=timit_test,
    tokenizer=processor.feature_extractor,
)

# Save processor
processor.save_pretrained(training_args.output_dir)

# Train model
trainer.train()

# Push to hub
trainer.push_to_hub()

# Save model and processor
model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
feature_extractor.save_pretrained(training_args.output_dir)
