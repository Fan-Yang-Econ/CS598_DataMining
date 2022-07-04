from CS598_DataMining.NLPUtility.CONFIG import get_or_load_sentence_transformer


def cal_sentence_embedding(x, round_to=3):
    return [round(i, round_to) for i in get_or_load_sentence_transformer().encode(x)]
