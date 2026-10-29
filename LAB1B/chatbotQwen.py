from transformers import AutoTokenizer, AutoModelForCausalLM


qwen_2_5 = "./models/qwen2.5-1.5b-instruct"
chatHistory = [{"prompt": "", "response": ""}]
systemPrompt = "You are a helpfull Assistant.\n"

def saveHistory(inputText, response):    
    chatHistory.append({            # Saves  each back and forth as a dictionary
        "prompt": inputText,
        "response": response
    })

def getHistory(length):
    if length <= 0:
        return ""                                                  
    history = "Use the conversation history to understand the user's latest question.\n\n" + "The following is the conversation historty: \n\n"

    for chat in chatHistory[-length:]:                              # extract promts and responses
        history += f"User: {chat['prompt']}\n"
        history += f"Assistant: {chat['response']}\n"

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

    enableHistory = bool(int(input("enableHistory 0/1: ")))
    while True:
        inputText = input("User input > ")

        if inputText.lower() == "quit":
            print("Exiting chatbot...")
            break
        if inputText.lower() == "debug":                  # for debug purposes, regarding context.
            print("History: \n\n",history)
            print("\n Prompt: \n", prompt)
            continue
        if inputText.lower() == "clear":
            chatHistory.clear()
            resopnseIndex = 0 
            history = "" 
            prompt = ""
            continue
        if not enableHistory:                                        # No conversation history
            prompt = inputText    
            response = generateResponse(prompt, tokenizer, model)
            print("Model response: ", response)
            continue
        
        history = getHistory((resopnseIndex))                                        # retrival of history
        prompt = systemPrompt + history + "Current user question: " + inputText      # model input string, with sysPrompt, context and user input
        response = generateResponse(prompt, tokenizer, model)
        print("Model response: ", response)
        
        resopnseIndex += 1
        saveHistory(inputText, response)

main()