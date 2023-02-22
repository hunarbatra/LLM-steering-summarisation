from analysis_pipeline.claims_extraction import claims_extraction_runner
from analysis_pipeline.prompt_validation import prompt_validation_runner
from analysis_pipeline.embedding_search import embedding_search_runner
from analysis_pipeline.rouge_score import rouge_score_runner

from utils import csv_export
from config import api_key

from pathlib import Path

import pandas as pd
import os


data_path = './outputs/csv_exports/'
all_files = os.listdir(data_path)


def check_file():
    path = Path('./analysis_pipeline/results.csv')
    return path.is_file()

def save_res(data):
    if not check_file():
        df = pd.DataFrame({'name': [], 'prompt_validation': [], 'embedding_validation': [], 'rouge': [], 'rouge_baseline': []})
        df.loc[len(df.index)] = data 
        csv_export(df, './analysis_pipeline/results')
    else:
        df = pd.read_csv('analysis_pipeline/results.csv')
        df.loc[len(df.index)] = data 
        csv_export(df, './analysis_pipeline/results')
    

for i, file in enumerate(all_files):
    if i < 7:
        continue
    df = pd.read_csv(data_path + file)
    data = df['text'][0]
    summary = df['summary'][0]
    claims_list, gpt3_summary = claims_extraction_runner(data)
    prompt_val_res = prompt_validation_runner(claims_list, summary)
    rouge_score = rouge_score_runner(data, summary)
    baseline_rouge_score = rouge_score_runner(data, gpt3_summary)
    embedding_val_res = embedding_search_runner(summary, claims_list, api_key)
    save_res([file.split('.')[0], prompt_val_res, embedding_val_res, rouge_score, baseline_rouge_score])
