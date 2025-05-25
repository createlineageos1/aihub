from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.eval()

def chat_with_gpt2(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chat_with_gpt2(user_input)
    print("GPT-2:", response)
