from transformers import AutoTokenizer, AutoModelForCausalLM


qwen_2_5 = "./models/qwen2.5-1.5b-instruct"
chatHistory = [{"prompt": "", "response": ""}]
systemPrompt = "You are a helpful Assistant. Keep your answers short.\n"

def saveHistory(inputText, response):    
    chatHistory.append({            # Saves  each back and forth as a dictionary
        "prompt": inputText,
        "response": response
    })

def getHistory(length):
    if length <= 0:
        return ""                                                  
    history = "Use the conversation history to understand the user's latest question.\n\n" + "The following is the conversation historty: \n\n"

    for i in range(len(chatHistory)-length, len(chatHistory)):                              # extract promts and responses
            history += f"User: {chatHistory[i]['prompt']}\n"
            history += f"Assistant: {chatHistory[i]['response']}\n"

    return history + "End of conversation history\n\n"

def generateResponse(inputText, tokenizer, model):

    messages = [{"role": "user", "content": inputText}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=128)
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    decodedOutputs = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return decodedOutputs

def main():
    tokenizer = AutoTokenizer.from_pretrained(qwen_2_5)
    model = AutoModelForCausalLM.from_pretrained(qwen_2_5)
    resopnseIndex = 0
    enableHistory = True

    print("Hi im your first chatbot in the ID1214 course!")
    print("Commands: \n quit - history - prompt - memory")
    enableHistory = bool(int(input("enableHistory 0/1: ")))

    while True:
        inputText = input("User input > ").lower()
        match inputText:
            case "quit":
                print("Exiting chatbot...")
                break
            case "history":
                print("History: \n\n",history)
                continue
            case "prompt":
                print("\n Prompt: \n", prompt)
                continue
            case "memory":
                enableHistory = bool(int(input("enableHistory 0/1: ")))
                continue

        if enableHistory:
            history = getHistory((resopnseIndex))                                        # retrival of history
            prompt = systemPrompt + history + "Current user question: " + inputText      # model input string, with sysPrompt, context and user input
            response = generateResponse(prompt, tokenizer, model)
            print("Model response: ", response)

            resopnseIndex += 1
            saveHistory(inputText, response)
        else:
            prompt = inputText    
            response = generateResponse(prompt, tokenizer, model)
            print("Model response: ", response)

main()