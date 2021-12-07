import sys, os, re, spacy
import pandas as pd
from textblob import TextBlob

def cleanup(text):
    paragraphs = text.split('\n')
    processed_text = ""
    for p in paragraphs:
        if len(p) < 15:
            continue
        doc = nlp(p)
        for token in doc:
            
            if not token.is_punct:
                replaced_tok = token.lemma_
                if token.lemma_ == '-PRON-':
                    replaced_tok = 'it'
                elif token.pos_ == 'PROPN':
                    replaced_tok = '[PROPN]'
                processed_text += (replaced_tok + " ")
            else:
                processed_text += (token.text + " ")
    processed_text = re.sub(r'"', "", processed_text)
    processed_text = re.sub(r"(good morning( everyone)*)", "", processed_text)
    processed_text = re.sub(r"(thank you)", "", processed_text)
    processed_text = re.sub(r"(thank(s)*)", "", processed_text)
    return processed_text

def prepare_data_frame(aggregated_text):
    dataFrame = pd.DataFrame(columns=["sentence", "subjectivity_score"])
    for text in aggregated_text:
        blob = TextBlob(text)
        for sentence_id, sentence in enumerate(blob.sentences):
            dataFrame.loc[sentence_id] = [sentence.raw, sentence.subjectivity]
    return dataFrame

def read_file():
    text_aggregate = []
    for root, dir, all_files in os.walk(os.getcwd()):
        for filename in all_files:
            if not filename.endswith(".txt") or filename.startswith("."):
                continue
            with open(os.path.join(root, filename ),'r', encoding='utf-8') as f:
                text = f.read()
            print("Processing", filename)
            # with open(directory + '/' + filename, 'w', encoding='utf-8') as p:
            #     p.write(tokenize(text))   
            text_aggregate.append(cleanup(text))
    return text_aggregate

        
if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    nlp = spacy.load('en_core_web_sm')
    aggregated = read_file()
    processed = prepare_data_frame(aggregated)
    processed.head()
    print(processed)