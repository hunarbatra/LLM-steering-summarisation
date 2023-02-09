from summarise import summary_runner, update_prompt, init_prompt, pred_tokens
from overview_extraction import overview_extraction_runner

from utils import read_file

# read data
data = read_file('data/test.txt')

generation_data: list[list] = []
ai_choice: list = []
curr_prompt: str = ''

for i in range(3):
    print('\nTimestep: {}\n'.format(i))
    if i == 0:
        overview = overview_extraction_runner(data)
        curr_prompt = init_prompt(use_overview_structure=True, overview_structure=overview)
    num_tokens = pred_tokens(curr_prompt, data)
    print(f'Number of prompt tokens used for current timestep: {num_tokens * 4}')
    curr_timestep_res = summary_runner(data, curr_prompt) # fetches 4 responses
    generation_data.append(curr_timestep_res)
    # ai_selected_res = ai_selection_runner(curr_timestep_res) # TODO
    # ai_choice.append(ai_selected_res)
    ai_selected_res = curr_timestep_res[0] # remove
    if i < 2:
        curr_prompt = update_prompt(curr_prompt, ai_selected_res)
    
print(generation_data)