from model import gpt3

from generate import generate_text, create_prompt, pred_tokens
from default_prompts import OVERVIEW_STRUCTURE_PROMPT

from utils import read_file

data = read_file('data/cyborgism.txt')

def init_overview_prompt():
    _prompt = """%s


    TEXT: {text}


    OVERVIEW STRUCTURE:""" % OVERVIEW_STRUCTURE_PROMPT
    return _prompt

def overview_extraction_runner(text, counter: int = 0):
    init_prompt = init_overview_prompt()
    llm = gpt3(max_tokens=512, temp=0.0)
    num_tokens = pred_tokens(init_prompt, text, llm)
    print(f'Number of prompt tokens used for overview structure extraction: {num_tokens}')
    prompt = create_prompt(init_prompt)
    overview_structure_text = generate_text(prompt, text, llm, None)
    if counter < 3:
        try:
            print('Overview Structure: ' + overview_structure_text)
        except: 
            overview_extraction_runner(text, counter + 1)
    else:
        print('error generating overview structure')
    return overview_structure_text

# overview_extraction_runner(data)
    