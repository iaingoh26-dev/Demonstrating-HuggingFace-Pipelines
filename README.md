# Hugging Face Pipelines Showcase

## Overview

This project demonstrates the use of Hugging Face Pipelines to perform a variety of AI tasks using pretrained transformer models. The notebook runs in Google Colab with GPU acceleration and showcases Natural Language Processing (NLP), image generation, and text-to-speech without training custom models.

## Features

- Sentiment Analysis
- Named Entity Recognition (NER)
- Question Answering
- Text Summarization
- Language Translation
- Zero-Shot Classification
- Text Generation
- Image Generation (Stable Diffusion XL Turbo)
- Text-to-Speech (SpeechT5)

## Technologies

- Python
- Hugging Face Transformers
- Hugging Face Diffusers
- Hugging Face Datasets
- PyTorch
- Google Colab
- CUDA

## Code Example

```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis", device="cuda")

result = sentiment("I'm super excited to be on the way to LLM mastery!")
print(result)
```

Example Output:

```text
[{'label': 'POSITIVE', 'score': 0.9998}]
```

## What I Learned

- Used Hugging Face Pipelines for multiple AI tasks with minimal code.
- Compared different pretrained models for the same task.
- Leveraged GPU acceleration for faster inference.
- Worked with image generation and text-to-speech models.
- Gained experience using transformer models for real-world AI applications.

## Challenges

- Configuring Hugging Face authentication using access tokens.
- Managing large model downloads and GPU memory limitations.
- Ensuring GPU availability in Google Colab.
- Understanding differences in outputs between pretrained models.

## Outcome

Successfully built a notebook demonstrating multiple AI capabilities using pretrained transformer models, gaining practical experience with modern NLP, image generation, and speech synthesis workflows.
