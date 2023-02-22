SUMMARISATION_PROMPT = "You are an intelligent summariser and a professional writer. You are given a piece of text. You will write a cohesive and detailed summary in 250 words covering all important aspects and key ideas of the text. Make sure that the generated text is not redundant, and it should be professionally summarised. Be sure to preserve important details and information of the text."

SUMMARISATION_STRUCTURE_PROMPT = "You are an intelligent summariser and a professional writer. You are given a piece of text and it's overview structure. You will use the text and overview structure to write a cohesive and detailed summary in 250 words step by step. You will cover all important aspects and key ideas presented in the text. Make sure that the generated text is not redundant, and it should be professionally summarised. Be sure to prioritise adding important details and information from the text in the summary."

OVERVIEW_STRUCTURE_PROMPT = "You are an intelligent and professional summariser. You will be given a piece of text, and you will generate a detailed overview structure for it covering all important key pointers and ideas in the outline."

AI_SELECTION_PROMPT = """You are given a piece of text, and four different partial summary option for it (1-4). You will select one summary options based on the selection criteria, and return the summary option number in the format: ANSWER: <number> and state the reasoning behind the selection by thinking step by step.
Selection Criteria: Select the summary which covers most of the important aspects of the text, is highly relevant to the text, and has the highest cosine similarity to the given text."""

CLAIMS_EXTRACTION = """You are given an alignment forum post. You will go over the post, and extract the 10 most important claims made in the post correctly. Be sure to avoid generating any redundant content, and include all important and relevant pointers in the list of claims."""

PROMPT_VALIDATION = """You are given a summary and a claim. You will assess if the given claim has been stated or referenced in the summary text clearly, and return your answer as True or False."""