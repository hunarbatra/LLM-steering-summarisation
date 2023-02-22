from rouge_score import rouge_scorer

def rouge_score_runner(post, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(post, summary)
    return scores['rouge1'][2]