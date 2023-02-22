from generate import generate_text, create_prompt, pred_tokens, gpt3_runner, chatgpt_runner

from default_prompts import PROMPT_VALIDATION

import time

def init_validation_prompt(claim, summary):
    _prompt = PROMPT_VALIDATION + f"""
    
    CLAIM: {claim}
    
    
    SUMMARY: {summary}

    
    Is the claim present in the summary, True or False:"""
    return _prompt

def prompt_validation_runner(claims_list, summary):
    validation_res = []
    for claim in claims_list:
        curr_prompt = init_validation_prompt(claim, summary)
        print(curr_prompt)
        # summary_validation = gpt3_runner(curr_prompt)
        summary_validation = chatgpt_runner(curr_prompt)
        time.sleep(2)
        res = summary_validation.strip().split(' ')[0].split('.')[0].lower()
        print(res)
        if res == 'true':
            validation_res.append(1)
        else:
            validation_res.append(0)
    print(f'Prompt Validation Score: {sum(validation_res) / len(validation_res)}; {validation_res}')
    return sum(validation_res) / len(validation_res)


    
        
        
        
        
        
        
        