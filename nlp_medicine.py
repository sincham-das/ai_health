from spacy.matcher import PhraseMatcher
import pandas as  pd
import spacy
from .image_process import result
import re
nlp=spacy.load("en_core_sci_scibert")

text=result
    
df=pd.read_csv("medicine_dictionary.csv")
medicine_list=df["medicine_name"].tolist()
matcher = PhraseMatcher(nlp.vocab)
patterns=[nlp.make_doc(med) for med in medicine_list]
matcher.add("MEDICINE", patterns)
doc=nlp(text)
matches=matcher(doc)
for match_id, start, end in matches:
    print(doc[start:end].text)
    
def extract_dosage(doc):
    text=doc.text
    dosage = re.findall(r'\d+\s?mg', text)
    frequency = re.findall(r'\d-\d-\d', text)
    duration = re.findall(r'\d+\s?(days|d)', text)
    
    return dosage, frequency, duration

