from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load pretrained transformer model

model_name = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Information

context = """
Pyhton is a programming language.
Artificial intelligence enables computers to perform intelligent tasks.
Machine Learning allows computer to learn from data.
"""

questions = [
    
      "What is Python?",
      "What is Artificial Intelligence?",
      "What is Machine Learning?",
      "Wahat is Java"
]


for question in questions:
    inputs = tokenizer(
             question,
             context, 
             return_tensors="pt"
            )
    
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits)
    
    answer = tokenizer.decode(
        inputs["input_ids"][0][start:end + 1],
        skip_special_tokens=True
    )
    print("\nQuestion:", question)
    print("Answer:", answer)
      
