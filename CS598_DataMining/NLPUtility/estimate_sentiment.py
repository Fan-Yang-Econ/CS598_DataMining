import numpy as np
from scipy.special import softmax

labels = ['negative', 'neutral', 'positive']


def estimate_sentiment(text) -> dict:
    """
    
    https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment?text=I+like+you.+I+love+you
    
    estimate_sentiment("Gun Talks Focus on Red-Flag Laws, Background Checks After Texas School Shooting")
    
    :param text:
    :return:

    np.round(0.023, 2)
    {'positive': 0.9783, 'neutral': 0.0189, 'negative': 0.0029}
    
    """
    from NLP_News.SpacyTools.CONFIG import get_or_load_sentiment_model
    model, tokenizer = get_or_load_sentiment_model()
    
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    
    r = {}
    for i in range(scores.shape[0]):
        _label = labels[ranking[i]]
        _score = scores[ranking[i]]
        r[_label] = float(_score)
    
    if r['positive'] > r['negative']:
        return np.round(r['positive'], 2)
    else:
        return -1 * np.round(r['positive'], 2)
