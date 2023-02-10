from model import gpt3
from summarise import generate_summary, create_prompt, pred_tokens

from default_prompts import AI_SELECTION_PROMPT

def init_selection_prompt(options):
    _prompt = """%s
    
    TEXT: {text}
    
    
    SUMMARY OPTIONS:
    """ % AI_SELECTION_PROMPT
    
    for i, option in enumerate(options):
        _prompt += str(i+1) + ". " + option + '\n'
        
    _prompt += '\n' + 'ANSWER:'
    return _prompt

def extract_option(result):
    print('AI Summary Selection result: ' + str(result) + '\n')
    result = result[0]
    try:
        result = int(result)
        print(f'AI selected option: {result}\n')
    except:
        print('error extracting answer from AIs feedback answer')
    return result

def ai_selection_runner(options, text, try_count: int = 0):
    # TODO: optimise token count
    init_prompt = init_selection_prompt(options)
    llm = gpt3(max_tokens=256, temp=0.0)
    num_tokens = pred_tokens(init_prompt, text, llm)
    print(f'Number of prompt tokens used for AI Selection: {num_tokens}\n')
    prompt = create_prompt(init_prompt)
    ai_selection_option = generate_summary(prompt, text, llm, None)
    final_option = extract_option(ai_selection_option)
    if final_option < 0 or final_option > 4:
        print('error in the selected value. trying again\n')
        try_count += 1
        if try_count > 3:
            print('error in AIs selected summary!!! selecting 1\n')
            final_option = 1
            return final_option - 1
        ai_selection_runner(options, text, try_count)
    return final_option - 1
    