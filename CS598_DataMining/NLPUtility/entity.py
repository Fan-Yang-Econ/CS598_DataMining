import re
import typing
import logging

import strbalance
import spacy

from CS598_DataMining.NLPUtility import get_or_load_spacy_model, get_or_load_en_ner

strbalance.Balance.STRAIGHT = ['"']


def get_entity_from_text(text):
    """
    text = "BERKSHIRE HATHAWAY INC. and Subsidiaries NOTES TO CONSOLIDATED FINANCIAL STATEMENTS December 31, 2017"
    result = get_entity_from_text(text)
    :param text:
    :return:

    [
        {'text': 'BERKSHIRE HATHAWAY INC.', 'start_char': 0, 'end_char': 23, 'label_': 'ORG'},
        {'text': 'December 31, 2017', 'start_char': 84, 'end_char': 101, 'label_': 'DATE'}
    ]

    """
    nlp = get_or_load_spacy_model()
    doc = nlp(text)
    
    result = []
    for i in doc.ents:
        result.append({'start_char': i.start_char, 'text': i.text, 'label_': i.label_})
    
    return result


def _is_ner_by_list_ner_idx_set(spacy_token, list_ner_idx_set):
    IS_NER = False
    set_idx = set(range(spacy_token.idx, len(spacy_token.text) + spacy_token.idx))
    for set_ner in list_ner_idx_set:
        if set_ner.intersection(set_idx):
            IS_NER = True
            break
    
    return IS_NER


# def better_capitalize_sentence(raw_: str):
#     """
#     In some news headline, every term is title-cased, which makes the noun-chunk algorithm hard to work.
#     So we need to first correct those title-cased terms first.
#     :param raw_:
#         raw_ = "Activision Blizzard Details Steps to Address Workplace Sexual Harassment Ahead of Planned Walkout"
#         x = "Over-Regulating Crypto Would Be ‘Disaster’: Q&A With Cam Harvey"
#         x_spacy = get_or_load_spacy_model()("Activision Blizzard details steps to address workplace sexual harassment ahead of planned walkout")
#     :return:
#     """
#     list_non_stop_items = [i for i in raw_.split() if i.lower() not in get_or_load_spacy_model().Defaults.stop_words]
#
#     if (
#             list_non_stop_items and
#             # This means most of terms are using the title case, so we need to transform them back!
#             sum([i.istitle() for i in list_non_stop_items]) / len(list_non_stop_items) > 0.7
#     ) or (
#             # This means at least one single quotation is there (we will transform it to double quotation)
#             "'" in raw_
#     ):
#
#         spacy_model = get_or_load_spacy_model()
#         # spacy_model.select_pipes(disable='ner')
#         # spacy_model.select_pipes(enable='ner')
#         # spacy_model.restore()
#         x_spacy = spacy_model(raw_)
#
#         list_ner = get_or_load_en_ner()(raw_)
#         list_ner_idx_set = [set(range(ner['start'], ner['end'])) for ner in list_ner]
#
#         new_text = ''
#         for i in x_spacy:
#             # i = x_spacy[0]
#             if _is_ner(i, list_ner_idx_set):
#                 new_i = i.text
#             elif i.text.istitle() and not i.text.isupper():
#                 new_i = i.text.lower()
#             else:
#                 # Some special forms such as `SPY`, or `iPhone`, which is not title-cased, so just keep its original form.
#                 new_i = i.text
#
#             # if it "Broker's", then "'s" dep_ is not `punct`, but `case`
#             if i.dep_ == 'punct' and i.text in ["'", "’", "‘"]:
#                 # Use double quote to replace single quotes for better noun-chunk identification
#                 new_i = '"'
#
#             if i.idx > 1 and raw_[(i.idx - 1)] == ' ':
#                 new_text += ' ' + new_i
#             else:
#                 new_text += new_i
#
#         return new_text
#
#     else:
#         return raw_


def better_capitalize_sentence(raw_: str):
    """
    In some news headline, every term is title-cased, which makes the noun-chunk algorithm hard to work.
    So we need to first correct those title-cased terms first.
    :param raw_:
        x = "Activision Blizzard Details Steps to Address Workplace Sexual Harassment Ahead of Planned Walkout"
        x = "Over-Regulating Crypto Would Be ‘Disaster’: Q&A With Cam Harvey"
        x = get_or_load_spacy_model()("Activision Blizzard details steps to address workplace sexual harassment ahead of planned walkout")
    :return:
    """
    list_non_stop_items = [i for i in raw_.split() if i.lower() not in get_or_load_spacy_model().Defaults.stop_words]
    
    if (
            list_non_stop_items and
            # This means most of terms are using the title case, so we need to transform them back!
            sum([i.istitle() for i in list_non_stop_items]) / len(list_non_stop_items) > 0.7
    ) or (
            # This means at least one single quotation is there (we will transform it to double quotation)
            "'" in raw_
    ):
        
        x_spacy = get_or_load_spacy_model()(raw_)
        
        if len([i for i in x_spacy if i.ent_type_ != '']) > len(x_spacy) * 0.8:
            list_ner = get_or_load_en_ner()(raw_)
            list_ner_idx_set = [set(range(ner['start'], ner['end'])) for ner in list_ner]
            
            def _is_token_a_ner(i):
                return _is_ner_by_list_ner_idx_set(i, list_ner_idx_set)
        
        else:
            def _is_token_a_ner(i):
                return i.ent_type_ in ['DATE', 'CARDINAL',
                                       'LANGUAGE', 'PRODUCT', 'LOC', 'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'LAW',
                                       'WORK_OF_ART']
        
        new_text = ''
        for i in x_spacy:
            # i = x_spacy[0]
            
            if _is_token_a_ner(i) or i.is_oov:
                new_i = i.text
            elif i.text.istitle() and not i.text.isupper():
                new_i = i.text.lower()
            else:
                # Some special forms such as `SPY`, or `iPhone`, which is not title-cased, so just keep its original form.
                new_i = i.text
            
            # if it "Broker's", then "'s" dep_ is not `punct`, but `case`
            if i.dep_ == 'punct' and i.text in ["'", "’", "‘"]:
                # Use double quote to replace single quotes for better noun-chunk identification
                new_i = '"'
            
            if i.idx > 1 and raw_[(i.idx - 1)] == ' ':
                new_text += ' ' + new_i
            else:
                new_text += new_i
        
        return new_text
    
    else:
        return raw_


MAX_LENGTH_OF_NOUN_CHUNK_TERMS = 10


def _token_is_stop(j):
    if j.is_stop or j.text.lower() in ['nearly', 'roughly', 'approximately']:
        return True
    else:
        return False


def cal_pos_score(pos_):
    if pos_ in ['AUX', 'DET', 'CONJ', 'CCONJ', 'PART', 'SCONJ', 'SYM', 'INTJ', 'SPACE']:
        return 0
    elif pos_ in ['ADP']:
        return 0.3
    elif pos_ in ['PUNCT', 'NOUN', 'X', None, 'NUM', 'PRON', 'PROPN']:
        return 1
    elif pos_ in ['ADV']:
        return 1.5
    elif pos_ in ['ADJ']:
        return 2
    elif pos_ in ['VERB']:
        return 5
    else:
        logging.info(f'Unknown pos_ `{pos_}`')
        return 0.8


def get_cleaned_noun_chunks(x: typing.Union[spacy.tokens.doc.Doc, str],
                            remove_special_entity_types=tuple(),
                            remove_stop_word_start=True,
                            simple_output=True
                            ):
    """

    :param x:
    :param remove_special_entity_types:
        ('DATE', 'TIME', 'MONEY', 'QUANTITY', 'PERCENT')
    :param remove_stop_word_start:
    :return:
    from pprint import pprint
    
    x_ = "Before Robinhood Markets came along, the retail investor complex controlled by firms like Fidelity, Wellington, Charles Schwab and E*Trade was sitting pretty. But the online trading platform run by Vlad Tenev has upended that, and like any good disruptor, made its fair share of enemies along the way. That’s come in the form of mountains of lawsuits, regulatory scrutiny, and distracting Washington gadflies that promise to reshape Robinhood. The backlash may be enough to leave the disruptor vulnerable to some disruption of its own"
    x=get_or_load_spacy_model()(x_)
    for i in x:
        print(i, _token_is_stop(i))
    get_cleaned_noun_chunks(x, remove_special_entity_types= ('DATE', 'TIME', 'MONEY', 'QUANTITY', 'PERCENT'))

    x = get_or_load_spacy_model()("Walmart is in talks to end quarterly bonuses, as it raises employees' hourly pay")
    pprint(get_cleaned_noun_chunks(x))
    
    x_ = "traders price in near-even odds of 75 basis-point fed hike in June"
    x=get_or_load_spacy_model()(x_)
    get_cleaned_noun_chunks(x, remove_special_entity_types= ('DATE', 'TIME', 'MONEY', 'QUANTITY', 'PERCENT'))

    
    """
    
    # x = df_feature['spacy_model'].iloc[4]
    
    remove_special_numeric_types = [i for i in remove_special_entity_types if i in ['MONEY', 'QUANTITY', 'PERCENT', 'CARDINAL', 'CARDINAL']]
    remove_special_non_numeric_types = [i for i in remove_special_entity_types if i not in remove_special_numeric_types]
    
    LIST_WORDS = []
    max_len = len(x.text) - 1
    
    for i_token in x:
        i_token.set_extension('noun_chunk_', default=False, force=True)
    for i_nc in x.noun_chunks:
        for i_token in i_nc:
            i_token._.noun_chunk_ = i_nc
    
    # Create new chunks
    list_new_noun_chunk = []
    for i_token in x:
        # Which subtree can be treated as noun_chunks?
        # 1) the root of the subtree is part of noun_chunk
        # 2) do you want to noun_chunk to be really long (with clause sentence? probably no, so delete the processed noun chunks that are 5 times of the original noun_vhunk)
        # print(i, i.dep_, i.pos_, i.head, list(i.subtree))
        for i_ in i_token.subtree:
            # print(list(i_token.subtree))
            if (i_._.noun_chunk_
                    and i_.head not in list(i_token.subtree)  # this token is the root of this subtree
                    and len(list(i_._.noun_chunk_)) <= len(list(i_token.subtree))  # this i.subtree contains all of tokens in this noun_chunk
            ):
                sub_tree_ = list(i_token.subtree)
                if 0 < len(sub_tree_) < MAX_LENGTH_OF_NOUN_CHUNK_TERMS:
                    i_nc_text = x.text[sub_tree_[0].idx:(sub_tree_[-1].idx + len(sub_tree_[-1]))]
                    
                    if strbalance.Balance(straight=True, german=True, cjk=True, math=True).is_unbalanced(i_nc_text) is None:
                        list_new_noun_chunk.append(sub_tree_)
    
    # Also pick up the noun_chunks suggested by Spacy.
    list_new_noun_chunk_text = [x.text[noun_chunk[0].idx:(noun_chunk[-1].idx + len(noun_chunk[-1]))] for noun_chunk in list_new_noun_chunk]
    
    for i_nc in x.noun_chunks:  # list(x.noun_chunks)
        # i_nc = list(x.noun_chunks)[2]
        i_nc_text = i_nc.text
        list_i_nc = list(i_nc)
        if i_nc_text not in list_new_noun_chunk_text:
            # while 1:
            #     if strbalance.Balance().is_unbalanced(i_nc_text) is not None and list_i_nc[-1].i + 1 < len(x):
            #         i_nc_text += x.text[i_nc_end_index:(len(x[list_i_nc[-1].i + 1]) + i_nc_end_index)]
            #         i_nc_end_index += len(x[list_i_nc[-1].i + 1])
            #         list_i_nc.append(x[list_i_nc[-1].i + 1])
            #     else:
            #         break
            if strbalance.Balance(straight=True, german=True, cjk=True, math=True).is_unbalanced(i_nc_text) is None:
                list_new_noun_chunk.append(list_i_nc)
    
    for noun_chunk in list_new_noun_chunk:
        # noun_chunk = list_new_noun_chunk[2]
        # if 'gamification' in i.text:
        #     raise KeyError()
        
        noun_chunk_text = x.text[
                          noun_chunk[0].idx:(noun_chunk[-1].idx + len(noun_chunk[-1]))
                          ]
        
        stop_words_idx_list = []
        _stop_words_relative_index_list = []
        # stop_words_text = []
        
        _DICT_TOKEN_INFO = {}
        
        for relative_index_j, j in enumerate(noun_chunk):
            # relative_index_j = 1; j = noun_chunk[relative_index_j]
            # if 'the' in j.text:
            #     raise Exception()
            logging.debug(f"{j}: {j.ent_type_}")
            
            dir(j)
            _DICT_TOKEN_INFO[j.text] = {
                'pos_': j.pos_,
                'dep_': j.dep_,
                'is_stop': j.is_stop,
                'score': cal_pos_score(j.pos_)
            }
            
            next_idx = min(max_len, j.idx + len(j.text))
            previous_idx = max(0, j.idx - 1)
            
            _INCLUDE_TO_STOP = False
            
            if (
                    # if the current item is a stop word
                    ((remove_stop_word_start and relative_index_j in [0, len(noun_chunk) - 1])
                     and _token_is_stop(j) and x.text[next_idx] in [' ']) or
                    # if the previous item is also stop word, then remove this one also
                    ((remove_stop_word_start and relative_index_j - 1 in _stop_words_relative_index_list) and _token_is_stop(j) and
                     x.text[next_idx] in [' ']) or
                    # `[75, -, basis, -, point, fed, hike]` allow to NOT DELETE the 75 here
                    j.ent_type_ in remove_special_non_numeric_types or
                    (
                            j.ent_type_ in remove_special_numeric_types and
                            not (
                                    relative_index_j < (len(noun_chunk) - 1) and noun_chunk[relative_index_j + 1].text == '-'
                            ) and
                            not (
                                    relative_index_j > 0 and noun_chunk[relative_index_j - 1].text == '-'
                            )
                            and j.text != '-'
                    ) or
                    (j.is_upper and (x.text[previous_idx] == '(' or x.text[next_idx] == ')'))  # remove upper words (normally tickers) in ()
            ):
                _INCLUDE_TO_STOP = True
            elif j.text == noun_chunk_text and _token_is_stop(j):  # if this noun_chunks itself is a stop word
                _INCLUDE_TO_STOP = True
            
            if _INCLUDE_TO_STOP:
                # print(j, j.ent_type_)
                stop_words_idx_list.extend(range(j.idx, j.idx + len(j.text)))
                _stop_words_relative_index_list.append(relative_index_j)
                # stop_words_text.append(j)
        
        if stop_words_idx_list:
            one_word = ''
            for index_ in [index_ for index_ in range(noun_chunk[0].idx, noun_chunk[-1].idx + len(noun_chunk[-1])) if
                           index_ not in stop_words_idx_list]:
                one_word += x.text[index_]
        else:
            one_word = noun_chunk_text
        
        # basic cleanings
        one_word = re.compile(' +').sub(' ', one_word).strip()
        one_word = re.compile(r'\[\d+]?$').sub('', one_word)  # remote the footnote
        one_word = re.compile(r'[=\-,:;?!&]+$').sub('', one_word).strip()  # remove the ending punctuations that do not have meanings.
        one_word = re.compile(r'\($').sub('', one_word).strip()  # remove the open-ended punctuations.
        one_word = re.compile(r'^\)').sub('', one_word).strip()  # remove the open-ended punctuations.
        one_word = re.compile(r'\([\W\d ]*\)').sub('', one_word)  # remove non-alphabet terms in parenthesis.
        one_word = re.compile(r' +').sub(' ', one_word).strip()
        
        if re.compile('[A-z]').findall(one_word):
            
            if one_word not in [i['_TEXT_'] for i in LIST_WORDS]:
                _DICT_TOKEN_INFO['_TEXT_'] = one_word
                LIST_WORDS.append(_DICT_TOKEN_INFO)
    
    if simple_output:
        return [i['_TEXT_'] for i in LIST_WORDS]
    else:
        return LIST_WORDS
