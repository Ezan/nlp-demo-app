import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import os
from .coreEngine import CoreEngine
from itertools import groupby
from operator import itemgetter
stopwords = stopwords.words('english')

class Classifier:
    
    def __init__(self, model_name, use_fast=True):
        self.model_path = os.path.join(os.getcwd(),"engine/model/{}".format(model_name))
        self.min_cosine_similarity_threshold = 0.10
        self._initQuestions()
        self.coreEngine = CoreEngine(model_path=self.model_path,use_fast=use_fast)

    def _initQuestions(self):
        self.questions = [
            "Find the mention of any experience gained that should be reviewed",
            "Find the mention of monetary amounts that should be reviewed",
            "Find the exaggerated statements made that should be reviewed",
            "Find the mention of opinions or hypothetical claims made that should be reviewed",
            "Find the mention of any development or progress that should be reviewed"
        ]

    def _clean_string(self, text):
        text = ''.join([word for word in text if word not in string.punctuation])
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stopwords])
        return text
    
    def _cosine_sim_vectors(self, vec1, vec2):
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]

    def __call__(self, store, isJson=False):
        result = []
        if not isJson:
            context = []
            store = sorted(store,key=itemgetter('slide_id'))
            for key, value in groupby(store,
                            key = itemgetter('slide_id')):
                text = ""
                for item in value:
                    if item['text'] == '' or item['text'] == ' ':
                        continue
                    text += (item['text'] + ". ")
                if text != '':
                    context.append({'slide_id':key, 'context':text})
            
            for cont in context:
                for question in self.questions:
                    answer = self.coreEngine(question, cont['context'])
                    result.append({'slide_id':cont['slide_id'], 'question':question, 'result': answer, 'context': cont})
        else:
            # vectorizer = CountVectorizer().fit_transform(list(map(self._clean_string, list(map(lambda x: x['true_answer'], store)) )))
            # true_vectors = vectorizer.toarray()
            for item in store:
                answers = self.coreEngine(item['question'], item['context'])
                for answer in answers:
                    cleaned_answer = self._clean_string(answer['answer'])
                    true_answer = self._clean_string(item['true_answer'])
                    vectorizer = CountVectorizer().fit_transform([cleaned_answer, true_answer])
                    vectors = vectorizer.toarray()
                    answer['cosine_similarity'] = self._cosine_sim_vectors(vectors[0], vectors[1])
                result += [{'context': item['context'], 'question':item['question'], 'true_answer':item['true_answer'],'result': answers}]    
            for item in result:
                item['similar_answers'] = list(map(lambda y: {'predicted_answer': y['answer'], 'cosine_similarity' : y['cosine_similarity']}, list(filter(lambda x: x['cosine_similarity'] >= self.min_cosine_similarity_threshold, item['result']))))
                item['dissimilar_answers'] = list(map(lambda y:{'other_answer': y['answer'], 'cosine_similarity' : y['cosine_similarity']}, list(filter(lambda x: x['cosine_similarity'] < self.min_cosine_similarity_threshold, item['result']))))
                del item['result']
        return result

if __name__ == "__main__":
    question = ["Find the mention of monetary amounts that should be reviewed"]
    context =  '''This document is being provided for preliminary discussion purposes only.  MRP Value Fund IB, LP (the “Fund”) does not intend to accept any subscription for, or otherwise sell or agree to sell, any securities until the subscriber or purchaser thereof has been provided with complete and final offering materials, including a final offering memorandum.  Only the Fund’s complete and final offering materials and the information contained therein or otherwise authorized thereby may be relied upon in connection with any offer of, sale of or decision to purchase any security related to the Fund.  Recipients are strongly advised to carefully review the Fund’s complete and final offering materials before making any such investment decision, including all disclosures in such materials regarding “risk factors”. 
Certain information contained in this document constitute “forward-looking statements,” which can be identified by the use of forward-looking terminology such as “may,” “will,” “should,” “expect,” “anticipate,” “project,” “estimate,” ‘forecast,” “intend,” “continue,” “target,” or “believe” or the negatives thereof or other variations thereon or comparable terminology.  Due to various risks and uncertainties, actual events or results or the actual performance may differ materially from those reflected or contemplated in such forward-looking statements.  Prospective investors should pay close attention to the assumptions underlying the analyses, forecasts and targets contained herein.  The analyses, forecasts, illustrations and targets contained in this document are based on assumptions believed to be reasonable in light of the information presently available.  Such assumptions (and the resulting analyses, forecasts, illustrations and targets) may require modification as additional information becomes available and as economic and market developments warrant.  Any such modification could be either favorable or adverse.  The forecasts, illustrations and targets have been prepared and are set out for illustrative purposes only, and no assurances can be made that they will materialize.  No assurance, representation or warranty is made by any person that any of the forecasts and targets will be achieved and no investor should rely on the forecasts and the targets.  Nothing contained in this document may be relied upon as a guarantee, promise, assurance or a representation as to the future.  Except as otherwise indicated, the information provided in this presentation is based on matters as they exist as of the date specified in the Document and not as of any future date, and will not be updated or otherwise revised to reflect information that subsequently becomes available or circumstances existing or changes occurring after the date hereof.  The views expressed in this Document are subject to change based on market and other conditions.
This document has been provided on a confidential basis for the sole use of the person to whom it is delivered by the Fund.  This document may not be distributed, reproduced or used without the consent of McGinty Road Partners, LP (“MRP” or the “Firm”) or for any purpose other than the evaluation of the Fund by the person to whom this presentation is delivered.  The recipient of this document agrees to keep the information contained herein confidential and not to disclose such information to any third party other than recipient’s attorneys, accountants, consultants and financial advisors (“Representatives”) on a need to know basis.  The recipient will inform its Representatives of the confidential nature of this document and direct such Representatives to treat the information contained herein confidentially and to use such information solely for its intended purpose.   Either upon a decision not to invest in the Fund, or at the request of MRP, each recipient shall return this document and/or delete all electronic copies in its possession.
The information contained herein has been prepared to assist the recipients in making their own evaluation of the Fund and does not purport to contain all information that the recipients may desire.  In all cases, interested parties should conduct their own investigation and analysis of the Fund, its business, prospects, results of operations and financial condition.  No party has made any kind of independent verification of any of the information set forth herein, including any statements with respect to projections or prospects of the Fund or the assumptions on which such statements are based, and does not undertake any obligation to do so.  MRP makes no representation or warranty, express or implied, as to the accuracy or completeness of this document or of the information contained herein and shall have no liability for the information contained in, or any omissions from, this presentation, or for any of the written, electronic or oral communications transmitted to the recipient in the course of the recipient’s own investigation and evaluation of the Fund.  This document speaks as of the date hereof, except where otherwise indicated, and MRP undertakes no obligation to update the information contained herein.
This document may contain information either prepared by or obtained from independent third party sources having no ownership of or managerial affiliation with the MRP or the Fund.  Any such information is believed to be reliable, but there can be no assurance as to the accuracy or completeness thereof.  Although MRP believes in good faith that all information and data provided by third-party sources that is either referred to or provided herein is reliable, MRP has not independently verified or ascertained, nor undertakes to verify or ascertain, any such data or information or the underlying economic assumptions relied upon by such sources.
The securities of the Fund have not been, nor will they be, registered under the United States securities laws, and such securities may not be, except in a transaction which does not violate such laws and the Fund’s governing documents, offered, sold, or transferred.  Recipients are not to construe the contents of this presentation as legal, business or tax advice.   Recipients must rely upon their own representatives, including legal counsel and accountants, as to any legal, tax, investment and other considerations.
Projected financial data for the investment opportunities discussed herein is based, in part, on assumptions made by the MRP team, which may not be borne out in the future.  Projections assume realization events at various times, with exit values typically based on the application of assumed capitalization rates, in addition to certain other disposition assumptions.  While the projected returns are based on assumptions which MRP currently believes to be reasonable under the circumstances, the actual return on the investments will depend on, among other factors, future operating results, market conditions and the value of the assets at the time of disposition, any related transaction costs, and the timing and manner of sale, all of which may differ from the assumptions and circumstances on which the projections are based.  Accordingly, the actual returns on investments may differ materially from the projected returns indicated herein. Further, past performance is not indicative of future results.
Unless otherwise noted, all internal rates of return (“IRRs”) are presented on a “gross” basis. “Projected Gross IRR” means an aggregate, compound, annual, internal rate of return on investments after deduction for servicing fees and transaction costs based on actual inflows and outflows from investments through the “as of date” indicated and projected inflows and outflows through the projected disposition date.  “Realized Gross IRR” means an aggregate, compound, annual, internal rate of return on investments after deduction for servicing fees and transaction costs based on actual inflows and outflows from investments through the “as of date” indicated.  Deal level performance figures are always presented as ‘gross’ figures. Fees are not charged on the deal level but rather on the Fund as a whole. “Projected Gross MOIC” means the sum of realized and unrealized proceeds after deduction of servicing fees and transactions costs based on actual inflows and outflows from investments through the “as of date” indicated and projected inflows and outflows through the projected disposition date divided by invested dollars.  A hypothetical illustration of the effect of incentive compensation on the returns of such investments may be requested from MRP.  The hypothetical example will illustrate the effect of any management fees, incentive compensation, “fund-level” expenses and other permitted expenses on returns, but will not be meant to indicate or reflect actual returns, fees or expenses, all of which may vary from the illustration.
“Projected Net IRR” means an aggregate, compound, annual, internal rate of return on investments after deduction for servicing fees, transaction costs, management fees, “fund-level” expenses and other permitted expenses based on actual inflows and outflows from investments through December 31, 2019 and projected inflows and outflows through the projected disposition date. “Projected Net MOIC” means the sum of realized and unrealized proceeds after deduction of servicing fees, transaction costs, management fees, “fund-level” expenses and other permitted expenses based on actual inflows and outflows from investments through December 31, 2019 and projected inflows and outflows through the projected disposition date divided by invested dollars.'''
    classifier = Classifier("roberta_second_iter",use_fast=True)
    textArrayWithShape = []
    textArrayWithShape.append({'slide_id':0,'shape_id':0, 'text' : ' '.join(context.split())})
    tofeed = (list(filter(lambda x: x['text'] is not None, textArrayWithShape)))
    result = classifier(tofeed)
    print(result)
