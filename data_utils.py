from transformers import GPT2Tokenizer

def collate_fn(batch, tokenizer, max_length=128):
    """Collate function for data loader"""
    texts = []
    for example in batch:
        text = f"question: {example['question']} answer: {example['answers']['text'][0]}"
        texts.append(text)
    encodings = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    return encodings 