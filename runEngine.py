import enum
from engine.helper.wordReader import DocReader
from engine.classifier.classifier import Classifier
from engine.helper.pdfReader import PDFXtraction
import os
import pandas as pd
import json

class Engine:
    class FileTypes(enum.Enum):
        text = 0
        word = 1
        ppt = 2
        json = 3
        pdf = 4

    def __init__(self, classifier_name="roberta_6_oct"):
        self.classifier = Classifier(classifier_name, use_fast=True)
        self.location = os.path.join(os.getcwd(),'uploads')
        self.output_dir = os.path.join(os.getcwd(), 'results')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
    
    def _processFile(self, filename):
        textArrayWithShape = []
        save_name = ""
        if (self.type == self.FileTypes.text):
            with open(os.path.join(self.location, filename), "r", encoding='utf-8') as f:
                textArrayWithShape.append({'slide_id':0,'shape_id':0, 'text' : ' '.join(f.read().split())})
            tofeed = (list(filter(lambda x: x['text'] is not None, textArrayWithShape)))
            result = self.classifier(tofeed)
            save_name = filename.split('.')[0] + '_result.xlsx'
            pd.DataFrame(result).to_excel(os.path.join(self.output_dir, save_name), sheet_name='result', index = False, engine='openpyxl')
        elif (self.type == self.FileTypes.json):
            with open(os.path.join(self.location, filename),"r", encoding='utf-8') as f:
                data = json.load(f)
                questions = list(map(lambda x: x['question'],data['Sheet1']))
                contexts = list(map(lambda x: x['context'],data['Sheet1']))
                true_answers =list(map(lambda x: x['text'],data['Sheet1']))
            results = []
            for question, context, true_answer in zip(questions, contexts, true_answers):
                textArrayWithShape.append({'slide_id':0,'shape_id':0, 'question': question, 'true_answer': true_answer, 'context' : ' '.join(context.split())})
                tofeed = (list(filter(lambda x: x['context'] is not None, textArrayWithShape)))
                result = self.classifier(tofeed, isJson=True)
                results.append(result)
            save_name = filename.split('.')[0] + '_result.xlsx'
            pd.DataFrame(result).to_excel(os.path.join(self.output_dir, save_name), sheet_name='result', index = False, engine='openpyxl')
                
        elif (self.type == self.FileTypes.word):
            reader = DocReader(os.path.join(self.location, filename))
            result = reader.extract()
            for text in result:
                textArrayWithShape.append({'slide_id':0,'shape_id':0, 'text' : ' '.join(text.split())})
            tofeed = (list(filter(lambda x: x['text'] is not None, textArrayWithShape)))
            result = self.classifier(tofeed)
            save_name = '.'.join(filename.split('.')[:-1]) + '_result.docx'
            reader.highlight(os.path.join(self.output_dir, save_name), result)
        elif (self.type == self.FileTypes.pdf):
            pdfExt = PDFXtraction(os.path.join(self.location, filename))
            los = pdfExt.extract()
            for text in los:
                textArrayWithShape.append({'slide_id':0,'shape_id':0, 'text' : ' '.join(text.split())})
                tofeed = (list(filter(lambda x: x['text'] is not None, textArrayWithShape)))
                result = self.classifier(tofeed)
                doc = pdfExt.highlight(result, text)
            save_name = '.'.join(filename.split('.')[:-1]) + '_result.pdf'
            doc.save(save_name)
        return save_name


    def checkFileType(self, filename):
        ext = filename.split('.')[-1]
        if ext == 'txt':
            self.type = self.FileTypes.text
        elif ext in ['doc', 'docx']:
            self.type = self.FileTypes.word
        elif ext in ['ppt','pptx']:
            self.type = self.FileTypes.ppt
        elif ext == 'json':
            self.type  = self.FileTypes.json
        elif ext in ['pdf']:
            self.type = self.FileTypes.pdf
        else:
            self.type = None
        return self._processFile(filename)