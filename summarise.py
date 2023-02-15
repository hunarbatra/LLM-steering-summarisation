from langchain import OpenAI, Prompt
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain

from model import gpt3
from default_prompts import SUMMARISATION_PROMPT, SUMMARISATION_STRUCTURE_PROMPT

import threading
import os

from config import api_key


os.environ["OPENAI_API_KEY"] = api_key
llm = gpt3()


def init_prompt(use_overview_structure: bool = False, overview_structure: str = None):
    if use_overview_structure == True:
        _prompt = SUMMARISATION_STRUCTURE_PROMPT
        _prompt +=  """

        {text}
        
        
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

def generate_summary(prompt, text, llm, curr_generation_data = None):
    mp_chain = MapReduceChain.from_params(llm, prompt, CharacterTextSplitter())
    try:
        response = mp_chain.run(text)
        if len(response) > 0:
            response = response.replace('\n','').strip()
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
        curr_thread = threading.Thread(target=generate_summary, args=(prompt, text, llm, curr_generation_data,))
        threads.append(curr_thread)
        curr_thread.start()
        
    for t in threads:
        t.join() # wait for all threads to stop before exiting the program
        
    return curr_generation_data