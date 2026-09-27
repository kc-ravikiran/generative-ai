from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load DialogSum dataset
dataset = load_dataset("kcravi/dialogsum")

# Load FLAN-T5 model
model_name = "google/flan-t5-base"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


# Create Few-Shot prompt
def make_prompt(example_indices_full, example_index_to_summarize):

    prompt = ""

    # Add multiple examples
    for index in example_indices_full:

        dialogue = dataset["test"][index]["dialogue"]
        summary = dataset["test"][index]["summary"]

        prompt += f"""
Dialogue:

{dialogue}

What was going on?
{summary}


"""

    # Add the new dialogue
    dialogue = dataset["test"][example_index_to_summarize]["dialogue"]

    prompt += f"""
Dialogue:

{dialogue}

What was going on?
"""

    return prompt


# Few-Shot examples from the lab
example_indices_full = [40, 80, 120]

# New dialogue to summarize
example_index_to_summarize = 200

# Create prompt
prompt = make_prompt(
    example_indices_full,
    example_index_to_summarize
)

# Generate summary
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

output = model.generate(
    inputs["input_ids"],
    max_new_tokens=50
)

summary = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)


# Display results
print("=" * 60)
print("FEW-SHOT PROMPTING")
print("=" * 60)

print("\nNumber of examples:", len(example_indices_full))

print("\nPrompt:")
print(prompt)

print("\nGenerated Summary:")
print(summary)
