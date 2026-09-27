from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --------------------------------------------------
# Load DialogSum dataset
# --------------------------------------------------

dataset = load_dataset("kcravi/dialogsum")

# --------------------------------------------------
# Load FLAN-T5
# --------------------------------------------------

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# --------------------------------------------------
# Select ONE example from the dataset
# --------------------------------------------------

example_index = 40

example_dialogue = dataset["test"][example_index]["dialogue"]
example_summary = dataset["test"][example_index]["summary"]

# --------------------------------------------------
# Select the NEW dialogue to summarize
# --------------------------------------------------

new_index = 200

new_dialogue = dataset["test"][new_index]["dialogue"]
human_summary = dataset["test"][new_index]["summary"]

# --------------------------------------------------
# ONE-SHOT PROMPT
# --------------------------------------------------

prompt = f"""
Dialogue:

{example_dialogue}

What was going on?
{example_summary}


Dialogue:

{new_dialogue}

What was going on?
"""

# --------------------------------------------------
# Tokenize
# --------------------------------------------------

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

# --------------------------------------------------
# Generate summary
# --------------------------------------------------

output = model.generate(
    inputs["input_ids"],
    max_new_tokens=50
)

# --------------------------------------------------
# Decode
# --------------------------------------------------

generated_summary = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

# --------------------------------------------------
# Display
# --------------------------------------------------

print("=" * 60)
print("ONE-SHOT INFERENCE")
print("=" * 60)

print("\nONE EXAMPLE GIVEN TO THE MODEL:")
print("-" * 60)
print(example_dialogue)

print("\nExample Summary:")
print(example_summary)

print("\n" + "=" * 60)

print("\nNEW DIALOGUE:")
print("-" * 60)
print(new_dialogue)

print("\nHuman Summary:")
print(human_summary)

print("\nModel Generated Summary:")
print(generated_summary)
