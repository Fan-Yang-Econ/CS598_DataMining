import spacy

nlp = spacy.load("en_core_web_md")

nlp('google').vector.sum()
sum([abs(i) for i in nlp('of').vector])
sum([abs(i) for i in nlp('google').vector])
