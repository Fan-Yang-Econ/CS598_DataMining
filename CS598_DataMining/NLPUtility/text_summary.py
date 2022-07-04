import re
import logging
import unicodedata

import spacy
from spacy.pipeline import Sentencizer

from PyHelpers import clean_messy_text
from CS598_DataMining.NLPUtility import get_or_load_spacy_model


def is_sentence_fuzzy_in_text(a_sentence, text, threshold=0.8):
    """
    If a sentence is in the text

    :param a_sentence:
        a_sentence ='A Forward-Looking Statements'
    :param text:
        text ='B Forward-Looking Statements'
    :param threshold:
    :return:
       assert is_sentence_fuzzy_in_text(a_sentence, text, 0.7)
       assert is_sentence_fuzzy_in_text(a_sentence, text, threshold=0.9) == False
       assert is_sentence_fuzzy_in_text(a_sentence='This is the Signature Page', text='This is the Signature Page to the Amendment and Supplementary Agreement to the NIO China Investment Agreement')

       is_sentence_fuzzy_in_text(a_sentence='Description of Business', text='Description')

    """
    try:
        if re.compile(a_sentence, flags=re.I).findall(text):
            # if a simple regex can match, then True
            return True
    except re.error:
        pass
    
    a_sentence_ = a_sentence.lower()
    text = text.lower()
    
    if a_sentence in text:
        return True
    
    a_sentence_list = re.compile(r'[ -\\\\]+').split(a_sentence_)
    
    text_ = unicodedata.normalize('NFKD', text)
    text_ = re.compile(r'\W+').sub(' ', text_)
    text_ = re.compile(r'\W+').sub(' ', text_)
    text_list = re.compile(r'[ -\\\\]+').split(text_)
    
    if a_sentence_list:
        words_in = [i for i in a_sentence_list if i in text_list]
        denominator = sum([len(part_sentence) for part_sentence in a_sentence_list])
        if denominator <= 1:
            return False
        return (sum([len(_one_word) for _one_word in words_in]) / denominator) > threshold
    else:
        return False


def get_context_str(token: "spacy.tokens.token.Token"):
    context_str = ''
    for position in range(-3, 3 + 1):
        try:
            context_str += token.nbor(position).text
        except IndexError as e:
            pass
    
    return context_str


def text_summary_to_sentences(pure_text, MAX_SUMMARY_SENTENCE=5, LIST_SUMMARY_TEXT_CANNOT_CONTAIN=None):
    # tr = pytextrank.TextRank()
    try:
        import pytextrank
        nlp = get_or_load_spacy_model()
        nlp.add_pipe('textrank', last=True)
    except Exception as e:
        if 'textrank' in str(e) and 'already exists' in str(e):
            pass
        else:
            raise e
    
    try:
        doc = nlp.make_doc(pure_text)  # create a Doc from raw text
        sentencizer = Sentencizer()
        sentencizer.punct_chars.add('\n')
        doc = sentencizer(doc)
        
        for count, token in enumerate(doc):
            if count > 0 and token.is_sent_start:
                context_str = get_context_str(token)
                if re.compile(r'No.\d', re.I).findall(context_str):
                    token.is_sent_start = False
        
        for name, proc in nlp.pipeline:  # iterate over components in order
            doc = proc(doc)
        
        list_summary_sentences = []
        
        for sent in doc._.textrank.summary(limit_phrases=1000, limit_sentences=MAX_SUMMARY_SENTENCE):
            
            if len(''.join([ent.text for ent in sent.ents])) > len(re.compile('[ .,:;]').sub('', sent.text)) * 0.5:
                logging.debug('More than 50% of text is as invalid entities, '
                              'so skip this sentence as summary sentence.')
                continue
            
            summary_text = clean_messy_text(full_text=sent.text, strip_ind=True, normalized_unicode=True)
            
            if re.compile('[a-z]', re.I).findall(summary_text):  # at least you need to have some text
                skip_ind = False
                
                if LIST_SUMMARY_TEXT_CANNOT_CONTAIN is not None:
                    for text_to_ignore in LIST_SUMMARY_TEXT_CANNOT_CONTAIN:
                        if skip_ind is False and is_sentence_fuzzy_in_text(a_sentence=text_to_ignore, text=summary_text):
                            logging.info(f'Ignore `{summary_text[0:200]}` as it matches {text_to_ignore}')
                            skip_ind = True
                
                if skip_ind is False:
                    list_summary_sentences.append(summary_text)
        
        logging.debug('=== list_summary_sentences ===')
        logging.debug([i for i in list_summary_sentences])
        
        return list_summary_sentences
    except ZeroDivisionError as e:
        logging.critical(str(e))
        return []
