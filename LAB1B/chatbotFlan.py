from transformers import T5Tokenizer, T5ForConditionalGeneration


flan_t5_small = "./models/model-flan-t5-small"
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
    input_ids = tokenizer(inputText, return_tensors="pt").input_ids             # Tokenize the text
    outputs = model.generate(input_ids, max_new_tokens=128)                     # Generate the output with the model
    decodedOutputs = tokenizer.decode(outputs[0], skip_special_tokens=True)     # Convert the tokenised output back to text

    return decodedOutputs

def main():
    tokenizer = T5Tokenizer.from_pretrained(flan_t5_small)
    model = T5ForConditionalGeneration.from_pretrained(flan_t5_small)
    resopnseIndex = 0
    enableHistory = True

    print("Hi im your first chatbot in the ID1214 course!")

    enableHistory = bool(int(input("enableHistory 0/1: ")))
    while True:
        inputText = input("User input > ")

        if inputText.lower() == "quit":
            print("Exiting chatbot...")
            break
        if inputText.lower() == "debug" and enableHistory.lower == "t":                  # for debug purposes, regarding context.
            print("History: \n\n",history)
            print("\n Prompt: \n", prompt)
            continue
        if inputText.lower() == "clear":
            chatHistory.clear()
            resopnseIndex = 0 
            history = "" 
            prompt = ""
            continue
        if enableHistory:                                        # No conversation history
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