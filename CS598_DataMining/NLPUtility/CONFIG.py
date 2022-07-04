import os

import psutil
import spacy

from CS598_DataMining.utility import create_folder

NLP_BEHAVIOR = {'spacy_model': {},
                'SentenceTransformer': {},
                'sentiment_model': {}
                }

if psutil.virtual_memory().total / 1024 / 1024 / 1024 < 4:
    NLP_BEHAVIOR['space_model_name'] = "en_core_web_sm"
else:
    NLP_BEHAVIOR['space_model_name'] = "en_core_web_md"


def get_cuda_or_cpu_device():
    import torch
    
    if NLP_BEHAVIOR.get('cuda_or_cpu_device') is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        NLP_BEHAVIOR['cuda_or_cpu_device'] = device
    else:
        device = NLP_BEHAVIOR['cuda_or_cpu_device']
    
    return device


def _load_huggingface_model_tokenizer(model_name, ModelClass, AutoTokenizer, revision=None):
    if model_name in NLP_BEHAVIOR:
        return NLP_BEHAVIOR[model_name]['model'], NLP_BEHAVIOR[model_name]['tokenizer']
    
    folder_model = create_folder(os.path.join('/tmp/', model_name, 'model'))
    folder_token = create_folder(os.path.join('/tmp/', model_name, 'token'))
    
    try:
        if revision:
            model = ModelClass.from_pretrained(folder_model, revision=revision)
        else:
            model = ModelClass.from_pretrained(folder_model)
    except Exception:
        model = ModelClass.from_pretrained(model_name)
        model.save_pretrained(folder_model)
    
    try:
        if revision:
            tokenizer = AutoTokenizer.from_pretrained(folder_token, revision=revision)
        else:
            tokenizer = AutoTokenizer.from_pretrained(folder_token)
    
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(folder_token)
    
    NLP_BEHAVIOR[model_name] = {}
    NLP_BEHAVIOR[model_name]['model'] = model
    NLP_BEHAVIOR[model_name]['tokenizer'] = tokenizer
    
    return model, tokenizer


def get_or_load_spacy_model(model_name=None) -> spacy.Language:
    if model_name is None:
        model_name = NLP_BEHAVIOR['space_model_name']
    
    model = NLP_BEHAVIOR['spacy_model'].get(model_name)
    
    if model is None:
        NLP_BEHAVIOR['spacy_model'][model_name] = spacy.load(model_name)
    
    return NLP_BEHAVIOR['spacy_model'][model_name]


def get_or_load_sentence_transformer(model_name: object = 'paraphrase-mpnet-base-v2') -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer
    
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    
    model = NLP_BEHAVIOR['SentenceTransformer'].get(model_name)
    if model is None:
        NLP_BEHAVIOR['SentenceTransformer'][model_name] = SentenceTransformer(model_name)
    
    return NLP_BEHAVIOR['SentenceTransformer'][model_name]


def get_or_load_en_generate_headline(model_name="Michau/t5-base-en-generate-headline"):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    
    model, tokenizer = _load_huggingface_model_tokenizer(model_name,
                                                         ModelClass=T5ForConditionalGeneration,
                                                         AutoTokenizer=T5Tokenizer,
                                                         )
    
    model = model.to(get_cuda_or_cpu_device())
    return model, tokenizer


def get_or_load_en_ner(model_name=r"dslim/bert-base-NER", revision='f7c2808a659015eeb8828f3f809a2f1be67a2446'):
    """
    https://huggingface.co/dslim/bert-base-NER
    
    :param model_name:
    :param revision:
    :return:
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline
    
    model, tokenizer = _load_huggingface_model_tokenizer(model_name,
                                                         ModelClass=AutoModelForTokenClassification,
                                                         AutoTokenizer=AutoTokenizer,
                                                         revision=revision)
    
    ner_model = pipeline("ner", model=model, tokenizer=tokenizer)
    
    return ner_model


def get_or_load_sentiment_model(model_name="cardiffnlp/twitter-roberta-base-sentiment"):
    """
    https://huggingface.co/dslim/bert-base-NER
    
    :param model:
    :param revision:
    :return:
    """
    
    model = NLP_BEHAVIOR['sentiment_model'].get(model_name)
    
    if model is None:
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer
        
        # Tasks:
        # emoji, emotion, hate, irony, offensive, sentiment
        # stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary
        
        ModelClass = AutoModelForSequenceClassification
        TokenizerClass = AutoTokenizer
        
        NLP_BEHAVIOR['sentiment_model'][model_name] = \
            _load_huggingface_model_tokenizer(model_name, AutoTokenizer=TokenizerClass,
                                              ModelClass=ModelClass)
    
    return NLP_BEHAVIOR['sentiment_model'][model_name]
