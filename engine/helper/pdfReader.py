from nltk.tokenize import word_tokenize
import fitz
import re 

class PDFXtraction:    
    
    def __init__(self,src, pos_list = [], pass_str_list = []):
        self.pos_list = pos_list
        self.pass_str_list = pass_str_list
        self.src = src   
        
    def extract(self):
        
        doc = fitz.open(self.src)
        pn = 0
        page_list = []
        pass_str = ""
        for page in doc:
            diction = page.getText("dict")
            page_list.append(pn)
            for items in diction['blocks']:
                num = items['number']
                for line in items['lines']:                    
                    for spn in line['spans']:
                        strt = len(pass_str)
                        pass_str = pass_str + spn['text'] +" "
                        info_dict = {'number' : num, 'text' : spn['text'], 'pos' : strt , 'page': pn }
                        tokens = word_tokenize(pass_str)
                        if(len(tokens)>=512):
                            if(tokens[-1] == '.' or tokens[-1] == '?' or tokens[-1] == '!'):
                                pass_dict = {'page':page_list,'text':pass_str}
                                self.pass_str_list.append(pass_dict)
                                pass_str = ""
                                page_list = [pn]
                        if(pn == doc.page_count-1 and items == diction['blocks'][-1] and line == items['lines'][-1] and spn == line['spans'][-1] ):
                            pass_dict = {'page':page_list,'text':pass_str}
                            self.pass_str_list.append(pass_dict)
                                
                self.pos_list.append(info_dict)
            pn+=1
            
        return self.pass_str_list


        
            
    def highlight(self, result, context):
        
        doc = fitz.open(self.src)

        for txt_blks in self.pass_str_list:
            if(context == txt_blks['text']):
                ld_pg = txt_blks['page']
    
        strt_crd = result['start']
        end_crd = result['end']
        phrse_hghlght = result['answer']
        found_num = []
    
        for item in self.pos_list:
            if(item['pos']>strt_crd and item['page'] in ld_pg):
                found_num.append(prev_item['number'])
            elif(item['pos']==strt_crd and item['page'] in ld_pg):
                found_num.append(item['number'])
            if(item['pos']>end_crd and item['page'] in ld_pg):
                found_num.append(prev_item['number'])
                break
            elif(item['pos']==end_crd):
                found_num.append(prev_item['number'])
                break
            prev_item = item
    

        for no in ld_pg:
            page = doc.loadPage(no)
            diction = page.getText("dict")
            blocky = diction['blocks']
            for items in blocky:
                for line in items['lines']:
                    for writ in line['spans']:            
                        if(phrse_hghlght in writ['text'] or writ['text'] in phrse_hghlght and items['number'] in found_num):
                            bndary = items['bbox']
                            boundary = fitz.Rect(bndary)
                            srch = page.search_for(phrse_hghlght)
                            if(len(srch)<=3):
                                highlight = page.addHighlightAnnot(srch)
                                highlight.setColors({"stroke":(1,0,1),"fill":(0.85,0.8,0.95) })
                                highlight.update()
                                continue
                            for res in srch:
                                res.normalize()
                                boundary.normalize()
                                if(res.contains(boundary) or boundary.contains(res) ):
                                    print("entered")
                                    highlight = page.addHighlightAnnot(res)
                                    highlight.setColors({"stroke":(1,0,1),"fill":(0.85,0.8,0.95) })
                                    highlight.update()
                                    
        return doc
        

# src = 'C:\\Users\\Asus\\Downloads\\Activity3.pdf'
# foract = PDFXtraction(src)
# print(foract.extract())
# foract.extract()
# context = foract.pass_str_list[0]
# result = [{'score': 0.21654856204986572, 'start': 2047, 'end': 2177, 'answer': 'Good leadership skills  are essential for the business to implement correct decision plan and grab the correct group of  consumers', 'cosine_similarity': 0.6454972243679029}]
# foract.highlight(resul, context['text'])

#res[1] == boundary[1] or res[3] == boundary[3] and items['number'] in found_num