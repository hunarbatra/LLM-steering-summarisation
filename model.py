from langchain import OpenAI

def gpt3(max_tokens: int = 80, temp: float = 0.8):
    return OpenAI(
            model_name = 'text-davinci-003', 
            # model_name = 'code-davinci-002',
            temperature = temp, # 0.0 
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            max_tokens = max_tokens, # 256 
            batch_size = 20, # batch size to use when passing multiple documents to generate
            verbose=True, # to print out
            n=1,
            best_of=1)