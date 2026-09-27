# One-Shot Prompting in Generative AI

## What Is One-Shot Prompting?

One-shot prompting is a form of in-context learning in which a prompt contains exactly one example of the task before the model is asked to handle a new input. The example shows the model the relationship between an input and the desired output, including the expected style or format.

For example:

```text
Dialogue: [example dialogue]
What was going on?
Two people are arranging a meeting.

Dialogue: [new dialogue]
What was going on?
```

The model uses the example as context to generate a summary for the new dialogue. The model's weights are not changed; this is prompting, not training or fine-tuning.

## Why One-Shot Prompting Matters

A single example can clarify a task when an instruction alone is ambiguous. It can demonstrate the desired output format, tone, or level of detail without requiring a dataset preparation and training process. It is a practical middle ground between zero-shot prompting, which provides no examples, and few-shot prompting, which provides several.

One-shot prompting is not guaranteed to improve every answer. The example needs to be correct and relevant, and the model can still produce an inaccurate result. The example also consumes tokens from the input context window and must be included for each request.

## One-Shot in the GenAI Toolkit

- **Zero-shot prompting:** Give the model an instruction but no example. This is simple and uses fewer input tokens, but may leave the intended format unclear.
- **One-shot prompting:** Give one example. This demonstrates a pattern with modest prompt length.
- **Few-shot prompting:** Give multiple examples. More examples can show variation or clarify edge cases, but use more context and may add noise if poorly chosen.
- **Fine-tuning:** Update model weights using training data. This can make a behavior persistent, but requires a training workflow and is a different approach from providing examples in a prompt.

## Parameters and Choices That Affect Results

One-shot prompting is a prompting strategy, not a model parameter. Its outcome is influenced by prompt design and generation settings:

- **Demonstration example:** Choose a correct example that resembles the target task. A poor or irrelevant example can mislead the model.
- **Example placement and order:** Put the demonstration before the target input, and keep a consistent input/output structure.
- **Instruction wording:** Make the requested task recognizable. In this program, each dialogue is followed by “What was going on?”
- **Output format:** The example summary signals that the answer should be a concise natural-language summary.
- **Context length:** The demonstration and target dialogue both consume input tokens and must fit the model's context limit.
- **`max_new_tokens`:** Sets the maximum number of newly generated tokens. This program sets it to `50`; it is a cap, not a guarantee of a complete or correct answer.
- **Temperature, top-k, and top-p:** These generation controls can influence randomness and output variation. This program does not pass them explicitly, so its model generation configuration or library defaults apply.
- **Model and tokenizer:** The model determines generation behavior; the tokenizer maps text to model tokens and back. The same prompt can behave differently with another model.

## Walkthrough of `One-Shot.py`

The program loads the DialogSum dataset and Google's FLAN-T5-base sequence-to-sequence model. It puts one dialogue and its known summary into a prompt as the demonstration, then appends a second dialogue without its summary. The generated answer is printed alongside the reference summary for comparison.

### 1. Import dependencies

```python
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
```

`load_dataset` loads data from the Hugging Face `datasets` library. `AutoModelForSeq2SeqLM` and `AutoTokenizer` load the model and the matching tokenizer through the Transformers library.

### 2. Load the dataset

```python
dataset = load_dataset("kcravi/dialogsum")
```

This loads the DialogSum dataset repository. The script selects rows from its `test` split using integer indices.

### 3. Load FLAN-T5 and its tokenizer

```python
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

`model_name` identifies the pretrained model. The tokenizer prepares input text for that model and decodes generated token IDs. `use_fast=True` requests the fast tokenizer implementation when available. `AutoModelForSeq2SeqLM` loads a model that generates a text sequence from an input sequence.

### 4. Select the one-shot example

```python
example_index = 40

example_dialogue = dataset["test"][example_index]["dialogue"]
example_summary = dataset["test"][example_index]["summary"]
```

The script selects dataset row 40 from the test split. Both the dialogue and its reference summary are retrieved. They form the single labeled demonstration in the prompt.

### 5. Select the new dialogue and its reference

```python
new_index = 200

new_dialogue = dataset["test"][new_index]["dialogue"]
human_summary = dataset["test"][new_index]["summary"]
```

Row 200 is the target dialogue. `new_dialogue` is included in the prompt. `human_summary` is retrieved as a reference for display, but it is not added to the prompt and is not given to the model as input.

### 6. Build the one-shot prompt

```python
prompt = f"""
Dialogue:

{example_dialogue}

What was going on?
{example_summary}


Dialogue:

{new_dialogue}

What was going on?
"""
```

The first dialogue-summary pair teaches the model the task pattern. The second dialogue is followed by the same question, but its summary is omitted. The model is expected to continue by generating that missing summary.

### 7. Tokenize and generate

```python
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

output = model.generate(
    inputs["input_ids"],
    max_new_tokens=50
)
```

`return_tensors="pt"` asks the tokenizer to return PyTorch tensors. The model receives the token IDs in `inputs["input_ids"]`. `max_new_tokens=50` limits the generated continuation to at most 50 new tokens.

### 8. Decode and display

```python
generated_summary = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)
```

The tokenizer converts the model's generated token IDs into readable text. `skip_special_tokens=True` removes special model tokens from the decoded string. The print statements then show the demonstration, target dialogue, human reference summary, and model-generated summary.

## Evaluation Note

The example and target are both selected from the dataset's `test` split. This is fine for a learning demonstration. For a rigorous evaluation on held-out test data, use a demonstration from a training or development split and reserve test rows for evaluation. The program prints a human reference and a generated answer but does not calculate a metric or automatically judge whether the generated summary is correct.
