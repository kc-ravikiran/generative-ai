# Foundation Question Answering

`Foundation.py` is a small extractive question-answering example using the Hugging Face model `distilbert-base-cased-distilled-squad`.

The script provides a short context about Python, artificial intelligence, and machine learning, then asks the model four questions. The answer is extracted from the supplied context.

## Requirements

- Python 3.8 or later
- `torch`
- `transformers`

The workspace virtual environment already contains the required packages. To install them in another environment:

```powershell
python -m pip install torch transformers
```

## Run

From this directory, run:

```powershell
.\.venv\Scripts\python.exe .\Foundation.py
```

On the first run, Hugging Face downloads and caches the tokenizer and model weights, so an internet connection is required. Later runs use the local cache.

## How It Works

1. The tokenizer encodes each question together with the context.
2. The question-answering model predicts the start and end token positions of the answer.
3. The script decodes those tokens and prints the result.

The final question concerns Java, which is not present in the context. Extractive question-answering models still select a span from the supplied text, so this answer is not a reliable indication that the context supports the question.