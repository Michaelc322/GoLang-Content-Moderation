from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model once when the server starts
tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert").eval()

MODEL_VERSION = "toxic-bert@v1"

@torch.no_grad()
def classify_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    score = probs[0][1].item()  # toxicity probability
    flagged = score > 0.8
    return score, flagged, MODEL_VERSION
