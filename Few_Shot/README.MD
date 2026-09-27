# Few-Shot Prompting in Generative AI

## What Is Few-Shot Prompting?

Few-shot prompting is a way to guide a generative AI model by including a small number of input-and-output examples in the prompt. The examples demonstrate the task and the kind of answer expected. The model then uses those examples as context when responding to a new input.

For example:

```text
Dialogue: [example dialogue]
What was going on?
Two people are arranging a meeting.

Dialogue: [new dialogue]
What was going on?
```

The model generates an answer for the new dialogue by following the pattern. Its weights are not updated; this is called **in-context learning**.

## Why It Matters

Few-shot prompting is useful when you want to adapt a model's behavior without training or fine-tuning it. Examples can teach the model the task, desired format, terminology, or tone. You can change the examples directly in the prompt, making this approach quick to experiment with.

There are tradeoffs:

- Examples use input tokens and take up part of the model's context window.
- Irrelevant or incorrect examples may lead to poor answers.
- The model can still make mistakes; examples do not guarantee correctness.
- The examples must be supplied again for each request because the model's weights are unchanged.

## Few-Shot Among GenAI Approaches

- **Zero-shot prompting:** Ask the model to perform a task without examples. It is concise, but the task or output format may be ambiguous.
- **One-shot prompting:** Provide one example. It can clarify the expected response pattern.
- **Few-shot prompting:** Provide several examples. This can make the task and format clearer, at the cost of more prompt tokens.
- **Fine-tuning:** Train a model further on examples so the change is reflected in its weights. This is different from few-shot prompting and generally requires a training process and data.

## Parameters and Choices That Affect Few-Shot Results

Few-shot prompting is a prompting technique rather than a single model parameter. Its results depend on choices in the prompt and on the model's generation settings:

- **Number of examples (shots):** More examples can clarify the pattern, but use more tokens. This program uses three.
- **Example selection:** Accurate, relevant examples are more useful than arbitrary ones. Examples similar to the new input may be especially helpful.
- **Example order:** The order can affect the response, so it may be worth testing different arrangements.
- **Example format:** Keep the examples consistent so the model can recognize the input/output pattern.
- **Task instruction:** State or demonstrate what the model should do. This program uses the question “What was going on?” after each dialogue.
- **Context length:** The prompt and generated answer must fit within the model's token limits.
- **Generation length:** `max_new_tokens` limits how many new tokens can be generated.
- **Generation variability:** Settings such as temperature and top-p can influence how varied the output is. This program does not set them explicitly, so the model's generation configuration or library defaults apply.

## Walkthrough of `few_shot.py`

The program uses Google's FLAN-T5-base model to summarize a dialogue from the DialogSum dataset. It builds a prompt with three dialogue-and-summary examples followed by a fourth dialogue that has no supplied summary. The model is asked to generate the missing summary.

### 1. Load the dataset

```python
dataset = load_dataset("kcravi/dialogsum")
```

This loads the DialogSum dataset. Later, the program reads examples from `dataset["test"]`.

### 2. Load the model and tokenizer

```python
model_name = "google/flan-t5-base"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
```

`AutoModelForSeq2SeqLM` loads a sequence-to-sequence model, which takes input text and generates output text. The tokenizer converts text to token IDs for the model and converts generated token IDs back to text. `use_fast=True` requests the fast tokenizer implementation when available.

### 3. Construct the prompt

```python
def make_prompt(example_indices_full, example_index_to_summarize):
```

The function accepts two arguments:

- `example_indices_full`: indices of the demonstration examples.
- `example_index_to_summarize`: the index of the dialogue for which a summary should be generated.

The loop retrieves each selected dialogue and its reference summary from the test split, then adds them to the prompt. After the examples, the function adds the target dialogue and the question, but not its summary. That leaves a continuation for the model to generate.

### 4. Select demonstrations and target

```python
example_indices_full = [40, 80, 120]
example_index_to_summarize = 200
```

The program uses dataset entries 40, 80, and 120 as its three examples. Entry 200 is the new dialogue to summarize. These are indices within the dataset's `test` split, not global dataset row numbers.

### 5. Tokenize the prompt

```python
inputs = tokenizer(prompt, return_tensors="pt")
```

The tokenizer turns the prompt into model input tensors. `return_tensors="pt"` requests PyTorch tensors.

### 6. Generate the summary

```python
output = model.generate(
    inputs["input_ids"],
    max_new_tokens=50
)
```

`model.generate` produces text based on the prompt. `max_new_tokens=50` allows up to 50 new tokens in the generated response; it is a maximum, not a guarantee of a complete or correct summary.

### 7. Decode and display the result

```python
summary = tokenizer.decode(output[0], skip_special_tokens=True)
```

This converts the generated token IDs back to readable text. `skip_special_tokens=True` omits model-specific control tokens. The remaining print statements display the number of examples, the prompt, and the generated summary.

## Evaluation Note

This script is a useful demonstration of few-shot prompting. However, it selects both its demonstrations and target from the `test` split. That is acceptable for a learning example, but if measuring model performance on a held-out test set, demonstrations should come from a training or separate development split so the test set remains reserved for evaluation.
