from model import gpt3
from generate import generate_text, create_prompt, pred_tokens, chatgpt_runner

from default_prompts import CLAIMS_EXTRACTION

def init_claims_prompt(summary):
    _prompt = f"""{CLAIMS_EXTRACTION}
    
    
    POST: {summary}
    
    
    LIST OF CLAIMS MADE IN THE POST:
    1."""
    return _prompt

def generate_summary(text):
    init_prompt = """Generate the summary of the following text.
    
    
    Text: {text}
    """
    llm = gpt3(max_tokens=250, temp=0.0)
    num_tokens = pred_tokens(init_prompt, text, llm)
    print(f'Number of prompt tokens used for naive summary generation: {num_tokens}')
    prompt = create_prompt(init_prompt)
    naive_summary = generate_text(prompt, text, llm, verbose=False)
    print('Naive Summary: ' + naive_summary)
    return naive_summary

def claims_extraction_runner(text: str) -> list:
    naive_summary = generate_summary(text)
    init_prompt = init_claims_prompt(naive_summary)
    claims = chatgpt_runner(init_prompt)
    claims_list = []
    for i, c in enumerate(claims.split('\n')):
        c = c.strip()
        curr_claim = c.split(' ')
        curr_claim = curr_claim[1:] if i > 0 else curr_claim
        curr_claim = ' '.join(curr_claim)
        claims_list.append(curr_claim)
    print('Extracted claims: ')
    for i, c in enumerate(claims_list):
        print(f'{i+1}. {c}\n')
    return claims_list, naive_summary