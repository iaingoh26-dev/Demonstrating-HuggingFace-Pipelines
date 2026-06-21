#  Hugging Face AI Pipelines Showcase

## Overview

This project demonstrates how to use Hugging Face's `pipeline()` API to perform a variety of Artificial Intelligence tasks using pre-trained transformer models. The notebook runs entirely in Google Colab with GPU acceleration and explores Natural Language Processing (NLP), image generation, and speech synthesis without the need to train models from scratch.

Rather than building custom machine learning models, this project focuses on understanding how to use state-of-the-art pretrained mod
---

## What This Project Does

The notebook demonstrates several AI capabilities using Hugging Face Pipelines:

### 1. Environment Setup

- Installs the required Hugging Face libraries.
- Verifies that Google Colab is connected to an NVIDIA Tesla T4 GPU.
- Imports all required Python packages.
- Authenticates with Hugging Face using a secure access token stored in Google Colab Secrets.

---

### 2. Sentiment Analysis

Uses pretrained transformer models to determine whether a sentence expresses positive or negative sentiment.

Example:

Input:
> "I'm super excited to be on the way to LLM mastery!"

Output:
- Positive
- Confidence score

The notebook also compares the default sentiment model with a multilingual BERT sentiment model to demonstrate how different pretrained models can produce different outputs.

---

### 3. Named Entity Recognition (NER)

Identifies important entities within a sentence such as:

- People
- Organizations
- Locations
- Products

Example:

Input:
> "Ed Donner teaches AI Engineering using Hugging Face."

Output:

- Person: Ed Donner
- Organization: Hugging Face

---

### 4. Question Answering

Uses a transformer model that answers questions using only the supplied context.

Example:

Question:
> What are Hugging Face pipelines?

Context:
> Pipelines are a high level API for inference...

The model extracts the most relevant answer directly from the provided text.

---

### 5. Text Summarization

Uses an abstractive summarization model to shorten long passages while preserving the main ideas.

Instead of copying text directly, the model generates a condensed version of the original content.

---

### 6. Language Translation

Demonstrates machine translation using pretrained transformer models.

Examples include:

- English → French
- English → Spanish

The notebook also shows how specifying different translation models changes the output.

---

### 7. Zero-Shot Classification

Classifies text into labels that were never used during training.

Example:

Input:
> "Hugging Face Transformers are amazing."

Candidate labels:

- Technology
- Sports
- Politics

The model predicts which label best matches the text.

---

### 8. Text Generation

Uses a language model to continue writing from a prompt.

Example:

Prompt:
> "If there's one thing I want you to remember..."

The model generates additional coherent text based on the supplied prompt.

---

### 9. AI Image Generation

Uses Stable Diffusion XL Turbo to generate images directly from text prompts.

Example prompt:

> "A class of students learning AI engineering in a vibrant pop-art style"

The model produces a unique AI-generated image using only the written description.

---

### 10. Text-to-Speech

Converts written text into realistic speech using Microsoft's SpeechT5 model.

The notebook also loads speaker embeddings so the generated speech has a consistent voice profile.

---

## Technologies Used

- Python
- Hugging Face Transformers
- Hugging Face Diffusers
- Hugging Face Datasets
- PyTorch
- Google Colab
- Stable Diffusion XL Turbo
- Microsoft SpeechT5
- CUDA GPU Acceleration

---

## What I Learned

Throughout this project I learned:

- How Hugging Face Pipelines simplify working with transformer models.
- How different pretrained models specialize in different AI tasks.
- How GPU acceleration significantly improves inference performance.
- How to authenticate securely with Hugging Face using access tokens.
- How diffusion models generate images from text prompts.
- How text-to-speech models combine pretrained models with speaker embeddings.
- How multiple AI tasks can be completed with only a few lines of code using pretrained models.

---

## Challenges Encountered

### GPU Availability

Google Colab does not always provide access to an NVIDIA Tesla T4 GPU, so hardware availability affected model performance.

---

### Large Model Downloads

Many transformer and diffusion models are several gigabytes in size, making the first execution significantly slower while model weights download.

---

### Hugging Face Authentication

Some models required a Hugging Face access token before they could be downloaded. Managing authentication securely through Google Colab Secrets was necessary.

---

### GPU Memory Limitations

Image generation and speech synthesis models consumed significantly more GPU memory than NLP models, requiring careful resource management.

---

### Comparing Different Models

Different pretrained models often produced different predictions for the same task. Experimenting with multiple models helped illustrate how model selection impacts output quality and accuracy.

---

## Project Outcome

Successfully demonstrated a broad range of AI capabilities using Hugging Face Pipelines, including:

- Sentiment Analysis
- Named Entity Recognition
- Question Answering
- Text Summarization
- Machine Translation
- Zero-Shot Classification
- Text Generation
- AI Image Generation
- Text-to-Speech

This project strengthened my understanding of transformer-based AI systems while providing practical experience using modern pretrained models for real-world inference tasks.
