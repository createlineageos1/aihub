from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

device = torch.device("cpu")
model.to(device)

def chat_with_phi(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    user_input = input("Sen: ")
    if user_input.lower() in ["exit", "quit"]: break
    output = chat_with_phi(user_input)
    print("Phi-2:", output)
