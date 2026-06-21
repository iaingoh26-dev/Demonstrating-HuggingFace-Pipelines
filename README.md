## HuggingFace Pipelines

A simple notebook that shows 9 AI superpowers using HuggingFace Pipelines.
Think of each pipeline like a **vending machine** — put something in, get a result out!

---

##  Requirements

```bash
pip install -q --upgrade datasets==3.6.0 transformers==4.57.6
```

You will also need:
- A **Google Colab** account
- A **HuggingFace account** with an API token
- A **Tesla T4 GPU** (free on Google Colab)

---

## 🔑 Setup

### 1. Check your GPU
```python
gpu_info = !nvidia-smi
# Should say: "Success - Connected to a T4"
```

### 2. Login to HuggingFace
```python
from huggingface_hub import login
from google.colab import userdata

hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)
```
> 💡 Add your HuggingFace token in the Colab left sidebar under the 🔑 key icon

### 3. Import everything
```python
import torch
from transformers import pipeline
from diffusers import DiffusionPipeline, AutoPipelineForText2Image
from datasets import load_dataset
import soundfile as sf
from IPython.display import Audio, display
```

---

## 🧠 The Golden Rule

Every single AI task in this notebook follows the same 3 steps:

```python
tool = pipeline("task-name")   # 1. Pick your tool
result = tool("your input")    # 2. Give it something
print(result)                  # 3. See the output
```

---

## 😊 1. Sentiment Analysis
> **What it does:** Reads text and decides if it's positive or negative

```python
# Simple version — positive or negative only
analyzer = pipeline("sentiment-analysis", device="cuda")
result = analyzer("I'm super excited to be learning AI!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

```python
# Better version — gives a star rating from 1 to 5
better = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device="cuda"
)
result = better("I should be more excited about this!")
print(result)
# Output: [{'label': '3 stars', 'score': 0.45}]
```

---

## 🏷️ 2. Named Entity Recognition (NER)
> **What it does:** Finds names of people, places, and organisations in text

```python
ner = pipeline("ner", device="cuda")
result = ner("AI Engineers are learning from HuggingFace in Google Colab from Ed Donner")

for entity in result:
    print(entity)
# Finds: HuggingFace (ORG), Google Colab (LOC), Ed Donner (PER)
```

---

## ❓ 3. Question Answering
> **What it does:** Reads a piece of text and answers questions about it

```python
question = "What are Hugging Face pipelines?"
context = "Pipelines are a high level API for inference of LLMs with common tasks"

qa = pipeline("question-answering", device="cuda")
result = qa(question=question, context=context)
print(result)
# Output: {'answer': 'a high level API for inference of LLMs'}
```
> 💡 The AI can only answer using the context you give it — like an open-book exam!

---

## 📝 4. Text Summarization
> **What it does:** Takes a long piece of text and makes it shorter

```python
summarizer = pipeline("summarization", device="cuda")

text = """
The Hugging Face transformers library is an incredibly versatile and powerful tool
for natural language processing (NLP). It allows users to perform a wide range of
tasks such as text classification, named entity recognition, and question answering.
It lowers the barrier to entry by providing Data Scientists with a convenient way
to work with transformer models.
"""

summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary[0]['summary_text'])
```

| Parameter | What it means |
|-----------|--------------|
| `max_length=50` | Summary can't be longer than 50 words |
| `min_length=25` | Summary can't be shorter than 25 words |
| `do_sample=False` | Give a consistent result every time |

---

## 🌍 5. Translation
> **What it does:** Translates text from one language to another

```python
# English to French
translator = pipeline("translation_en_to_fr", device="cuda")
result = translator("The Data Scientists were amazed by the HuggingFace pipeline API.")
print(result[0]['translation_text'])
```

```python
# English to Spanish — with a specific model
translator = pipeline(
    "translation_en_to_es",
    model="Helsinki-NLP/opus-mt-en-es",   # specific model for better accuracy
    device="cuda"
)
result = translator("The Data Scientists were amazed by the HuggingFace pipeline API.")
print(result[0]['translation_text'])
```
> 💡 Find all translation models here:
> https://huggingface.co/models?pipeline_tag=translation&sort=trending

---

## 🗂️ 6. Zero-Shot Classification
> **What it does:** Sorts text into categories — even ones it has never seen before!

```python
classifier = pipeline("zero-shot-classification", device="cuda")

result = classifier(
    "Hugging Face's Transformers library is amazing!",
    candidate_labels=["technology", "sports", "politics"]   # you choose the categories
)
print(result)
# Output: technology wins with high confidence ✅
```
> 💡 You give it the labels — it figures out which one fits best

---

## ✍️ 7. Text Generation
> **What it does:** Continues writing from where you leave off

```python
generator = pipeline("text-generation", device="cuda")

result = generator("If there's one thing I want you to remember about HuggingFace pipelines, it's")
print(result[0]['generated_text'])
# The AI finishes the sentence for you!
```

---

## 🎨 8. Image Generation
> **What it does:** Creates
