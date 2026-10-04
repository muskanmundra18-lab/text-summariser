# Text Summariser

A transformer-based text summarization application using a fine-tuned **T5** model and a **FastAPI** backend with a simple HTML/CSS frontend.

## Overview

The application accepts dialogue/text, preprocesses it, tokenizes it with a T5 tokenizer, generates a summary, and returns the result through an API.

```text
Text / Dialogue
      ↓
Cleaning
      ↓
T5 Tokenizer
      ↓
T5 Model
      ↓
Beam Search Generation
      ↓
Summary
```

## Features

- Text/dialogue input
- Text cleaning and normalization
- T5-based sequence-to-sequence summarization
- FastAPI REST endpoint
- HTML/CSS frontend
- Automatic CPU/CUDA/MPS device selection
- Beam-search decoding

The current implementation limits the input to 512 tokens and generates summaries up to 150 tokens.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- T5
- FastAPI
- Pydantic
- Jinja2
- HTML
- CSS

## Repository Structure

```text
text-summariser/
├── app.py
├── index.html
├── style.css
├── text_summary.ipynb
└── README.md
```

## Getting Started

### Install dependencies

```bash
pip install fastapi uvicorn torch transformers pydantic jinja2
```

### Model

The API expects the trained T5 model and tokenizer under:

```text
saved_summary_model/
```

### Run the API

```bash
uvicorn app:app --reload
```

Open the local application in your browser.

## API

### POST `/summarize/`

Example request:

```json
{
  "dialogue": "Your text or dialogue goes here."
}
```

Example response:

```json
{
  "summary": "Generated summary..."
}
```

## Implementation Details

The application uses:

- `T5ForConditionalGeneration`
- `T5Tokenizer`
- `max_length=512` for input
- `max_length=150` for generated output
- Beam search with `num_beams=4`

## Future Improvements

- Add a configurable summary length
- Support longer documents through chunking
- Add model evaluation with ROUGE
- Add file upload support
- Add summary history
- Containerize the API with Docker

## Author

**Muskan Mundra**
