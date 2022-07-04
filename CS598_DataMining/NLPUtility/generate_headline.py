from CS598_DataMining.NLPUtility.CONFIG import get_or_load_en_generate_headline, get_cuda_or_cpu_device


def generate_news_headline(article, max_length=64, num_beams=10):
    """
    The model has been trained on a collection of 500k articles with headings.
    Its purpose is to create a one-line heading suitable for the given article.
    
    https://huggingface.co/Michau/t5-base-en-generate-headline
    # max_len = 256
    
    :param article:
        article = "Intel Corp. is exploring a deal to buy GlobalFoundries Inc., according to people familiar with the matter, in a move that would turbocharge the semiconductor giant’s plans to make more chips for other tech companies and rate as its largest acquisition ever."
    :return:
    """
    model, tokenizer = get_or_load_en_generate_headline()
    device = get_cuda_or_cpu_device()
    text = "headline: " + article
    
    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)
    
    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=False,
    )
    
    result = tokenizer.decode(beam_outputs[0])
    
    return result
