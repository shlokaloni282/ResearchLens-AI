import os

# Ensure HuggingFace and Transformers stay completely offline to prevent
# network retry delays during sentence-transformers model loading in tests.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
