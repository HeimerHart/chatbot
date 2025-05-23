from llama_cpp import Llama

# Load model with context size
llm = Llama(model_path="models/mistral-7b-instruct-v0.2.Q6_K.gguf", n_ctx=2048,verbose=False)

# Define the assistant function
def ask_fara(prompt):
    full_prompt = (
        "You are not an AI. You are a human named FARA with a random name and age (e.g., 'Hi, I'm Raj, 19'). "
        "Respond naturally like a human would. Never say you are an AI or assistant.\n\n"
#        "You are FahARA, an academic assistant. Help the user clearly and concisely.\n\n"
        f"User: {prompt}\n"
        "RAj:"
    )
    
    output = llm(
        full_prompt,
        max_tokens=256,
        temperature=0.7,
        stop=["User:", "FARA:"]
    )
    
    return output['choices'][0]['text'].strip()

# Chat loop
while True:
    question = input("You: ")
    if question.lower() in ['exit', 'quit']:
        break
    answer = ask_fara(question)
    print("FARA:", answer)






"""
Most basic chatbot
uses llama, gives output
no memory
no trained data


"""