import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
import json
import os

def load_squad():
    print("Loading SQuAD dataset...")
    # Load the SQuAD dataset
    dataset = load_dataset("rajpurkar/squad")
    # Get the train and validation splits
    train_data = dataset['train']
    val_data = dataset['validation']
    
    return train_data, val_data



if __name__ == "__main__":
    # Load and split the dataset
    train_data, val_data = load_squad()
    print(f"Train set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")