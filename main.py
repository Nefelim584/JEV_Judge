import os
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-large", device_map="auto")