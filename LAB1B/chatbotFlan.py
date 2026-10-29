from transformers import T5Tokenizer, T5ForConditionalGeneration


flan_t5_small = "./models/model-flan-t5-small"
chatHistory = [{"prompt": "", "response": ""}]
systemPrompt = "You are a helpfull Assistant.\n"

def saveHistory(inputText, response):    
    chatHistory.append({            # appends each back and forth to the chat history
        "prompt": inputText,
        "response": response
    })

def getHistory(length):
    if length <= 0:
        return ""                                                  
    history = "Use the conversation history to understand the user's latest question.\n\n" + "The following is the conversation historty: \n\n"
    # extract promts and responses
    for i in range(len(chatHistory)-length, len(chatHistory)):                              # extract promts and responses
        history += f"User: {chatHistory[i]['prompt']}\n"
        history += f"Assistant: {chatHistory[i]['response']}\n"

    return history + "End of conversation history\n\n"

def generateResponse(inputText, tokenizer, model):
    input_ids = tokenizer(inputText, return_tensors="pt").input_ids             # Tokenize the text input
    outputs = model.generate(input_ids, max_new_tokens=128)                     # Generate the output with the model, with the tokenized text as input
    decodedOutputs = tokenizer.decode(outputs[0], skip_special_tokens=True)     # Convert the tokenised output back to text

    return decodedOutputs

def main():
    tokenizer = T5Tokenizer.from_pretrained(flan_t5_small)
    model = T5ForConditionalGeneration.from_pretrained(flan_t5_small)
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
            prompt = systemPrompt + history + "Current user question: " + inputText      # Building of the prompt string: sysPrompt, history, quert etc.
            response = generateResponse(prompt, tokenizer, model)
            print("Model response: ", response)

            resopnseIndex += 1
            saveHistory(inputText, response)
        else:
            prompt = systemPrompt + inputText                                           # Building of the prompt string excluding history
            response = generateResponse(prompt, tokenizer, model)
            print("Model response: ", response)
main()