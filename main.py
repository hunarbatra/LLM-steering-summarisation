from summarise import summary_runner, update_prompt, init_prompt, pred_tokens
from overview_extraction import overview_extraction_runner
from ai_feedback import ai_selection_runner

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
    ai_selected_res = ai_selection_runner(curr_timestep_res, data) 
    ai_choice.append(ai_selected_res)
    if i < 2: # update prompt with the selected generated response
        curr_prompt = update_prompt(curr_prompt, curr_timestep_res[ai_selected_res])
        
final_summary = ''

for i, timestep in enumerate(generation_data):
    final_summary += timestep[ai_choice[i]]
    
print(generation_data)

print('\nFinal AI-selected Summary: ' + final_summary)