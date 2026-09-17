# Transformer Architecture

A simple Jupyter Notebook walkthrough showing how a pretrained Transformer language model processes text and predicts the next token.

## Overview

This notebook uses Hugging Face Transformers and PyTorch with the pretrained **DistilGPT-2** model. It demonstrates the main stages of next-token prediction:

```text
Input Text
    ↓
Tokenization
    ↓
Token IDs
    ↓
Embeddings
    ↓
Position Information
    ↓
Transformer Processing
    ↓
Logits
    ↓
Softmax
    ↓
Top-5 Probabilities
    ↓
Select Next Token
    ↓
Token ID → Text
```

## Requirements

- Python
- Jupyter Notebook / JupyterLab
- PyTorch
- Hugging Face Transformers

The notebook installs the required packages with:

```python
%pip install transformers torch
```

If `transformers` is not installed, you may see:

```text
ModuleNotFoundError: No module named 'transformers'
```

Run the installation cell and restart the notebook kernel/runtime if necessary.

## Model and Tokenizer

The notebook loads:

- **Tokenizer:** `distilgpt2`
- **Language model:** `distilgpt2`
- **Libraries:** `transformers`, `torch`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
```

The pretrained model is used to generate the scores needed for next-token prediction.

## Walkthrough

### 1. Input Text

The example input is:

```text
The sun rises in the
```

This text is passed to the tokenizer.

### 2. Tokenization

The tokenizer converts the input text into tokens:

```python
inputs = tokenizer(text, return_tensors="pt")
```

The notebook also displays the individual tokens using:

```python
tokenizer.convert_ids_to_tokens(token_ids[0])
```

### 3. Token IDs

Each token is represented by a numerical token ID.

The notebook prints the mapping:

```text
Token → Token ID
```

This illustrates how text is converted into numerical data that the model can process.

### 4. Embeddings

The token IDs are converted into numerical vectors using the model's token embedding table:

```python
with torch.no_grad():
    embeddings = model.transformer.wte(token_ids)
```

The notebook prints the embedding shape and the embedding vector for the first token.

The key idea is that each token is represented as a numerical vector before being processed by the Transformer.

### 5. Position Information

The notebook lists each token's position:

```text
Position 0 → ...
Position 1 → ...
Position 2 → ...
```

It explains that position information allows the Transformer to understand the order of tokens.

### 6. Transformer Processing

The input is passed through the pretrained model:

```python
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
```

The notebook presents the Transformer processing conceptually as:

```text
Embedding + Position
        ↓
Self-Attention
        ↓
Feed-Forward Network
        ↓
Transformer Layers
```

### 7. Logits

The model produces logits for the vocabulary:

```python
next_token_logits = logits[:, -1, :]
```

These are raw scores for possible next tokens.

The notebook displays the shape of both the complete logits tensor and the logits corresponding to the last input token.

### 8. Softmax

The raw logits are converted into probabilities:

```python
probabilities = torch.softmax(next_token_logits, dim=-1)
```

The resulting values represent the model's probability distribution over possible next tokens.

### 9. Top-5 Next-Token Probabilities

The notebook obtains the five tokens with the highest probabilities:

```python
top_probs, top_token_ids = torch.topk(probabilities, 5)
```

It then decodes each token ID back into text and prints its probability.

### 10. Select the Next Token

The notebook selects the token with the highest probability:

```python
next_token_id = torch.argmax(probabilities, dim=-1)
```

This is the greedy next-token selection step.

### 11. Token ID → Text

Finally, the selected token ID is decoded back into text:

```python
next_token = tokenizer.decode(next_token_id)
```

The predicted next token is printed as the final prediction.

## Key Concepts Demonstrated

| Concept | Purpose |
|---|---|
| Tokenization | Converts text into model-compatible tokens |
| Token IDs | Numerical representation of tokens |
| Embeddings | Converts tokens into numerical vectors |
| Position Information | Represents token order |
| Self-Attention | Allows tokens to interact with contextual information |
| Transformer Layers | Process the input representation |
| Logits | Raw scores for possible vocabulary tokens |
| Softmax | Converts scores into probabilities |
| Top-k | Identifies the highest-probability candidates |
| Argmax | Selects the highest-probability token |
| Decoding | Converts a token ID back into text |

## Running the Notebook

1. Open `Transformer_Architecture.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or another compatible notebook environment.
2. Run the installation cell:
   ```python
   %pip install transformers torch
   ```
3. Run the remaining cells from top to bottom.
4. The first model download may take some time because the pretrained model needs to be downloaded.
5. Inspect the printed output at each step to follow the transformation from text to a predicted next token.

## Important Note

The notebook is primarily an educational walkthrough. It exposes selected intermediate values to illustrate the Transformer pipeline rather than implementing a Transformer architecture from scratch.

## Files

```text
.
├── Transformer_Architecture.ipynb
└── README.md
```
