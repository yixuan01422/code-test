from transformers import GPT2Tokenizer

def collate_fn(batch, tokenizer, max_length=128):
    """Collate function for data loader"""
    texts = []
    answer_texts = []
    for example in batch:
        text = f"question: {example['question']} answer: {example['answers']['text'][0]}"
        texts.append(text)
        answer_texts.append(example['answers']['text'][0])
    encodings = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    encodings['answer_text'] = answer_texts
    return encodings 