import openai
from revChatGPT.V1 import Chatbot

from langchain import OpenAI, Prompt
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain

from model import gpt3
from default_prompts import SUMMARISATION_PROMPT, SUMMARISATION_STRUCTURE_PROMPT

import threading
import os

from config import api_key, chatgpt_email, chatgpt_pass, chatgpt_access_token, chatgpt_access_token2


os.environ["OPENAI_API_KEY"] = api_key
llm = gpt3()


def init_prompt(use_overview_structure: bool = False, overview_structure: str = None):
    if use_overview_structure == True:
        _prompt = SUMMARISATION_STRUCTURE_PROMPT
        _prompt +=  """

        TEXT: {text}
        
        
        OVERVIEW STRUCTURE: %s


        SUMMARY:""" % overview_structure
    else: 
         _prompt =  """%s

        {text}


        SUMMARY:""" % SUMMARISATION_PROMPT
    return _prompt
        
def create_prompt(p):
    _prompt = Prompt(template=p, input_variables=["text"])
    return _prompt

def update_prompt(p, selected_response):
    p += selected_response
    return p
    
def pred_tokens(p, data, llm = llm):
    # use tiktoken to predict number of tokens that will be used
    num_tokens = OpenAI.get_num_tokens(llm, p.replace('{text}', data))
    return num_tokens

def generate_text(prompt, text, llm, curr_generation_data = None, preprocess: bool = True, verbose: bool = True):
    mp_chain = MapReduceChain.from_params(llm, prompt, CharacterTextSplitter())
    try:
        response = mp_chain.run(text)
        if len(response) > 0:
            response = response.replace('\n','').strip() if preprocess else response.strip()
            if verbose:
                print(response + '\n')
            if curr_generation_data != None: # multiverse text generation
                curr_generation_data.append(response)
            else: # overview structure generation
                return response
        else:
            print('empty response generated')
    except:
        print('err generating response') 
    
def summary_runner(text, curr_prompt):
    threads: list = []
    curr_generation_data: list = []
    prompt = create_prompt(curr_prompt)
    for i in range(4):
        curr_thread = threading.Thread(target=generate_text, args=(prompt, text, llm, curr_generation_data,))
        threads.append(curr_thread)
        curr_thread.start()
        
    for t in threads:
        t.join() # wait for all threads to stop before exiting the program
        
    return curr_generation_data

def gpt3_runner(prompt, max_tokens: int = 50):
    completion = openai.Completion()
    response = completion.create(engine="text-davinci-003",
                                 prompt=prompt,
                                 temperature=0.0,
                                 max_tokens=max_tokens, 
                                 top_p=1.0,
                                 frequency_penalty=0.0,
                                 presence_penalty=0)
    answer = response.choices[0].text.strip()
    return answer

def chatgpt_runner(prompt):
    chatbot = Chatbot(config={
        # "email": chatgpt_email,
        # "password": chatgpt_pass
        "access_token": chatgpt_access_token2
    })
    response = ""
    for data in chatbot.ask(prompt):
        response = data["message"]
    return response