import re
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
import argparse
import os
import json
import numpy as np

class Eval:
    def __init__(self):
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]
        
    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText
    
    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip('\'')
        answer = answer.strip('\"')
        answer = answer.strip().lower()
        return answer

    def evaluate_rouge(self,preds):
        rouge = Rouge()
        cider = Cider()
        bleu = Bleu(4)
        acc = {'f': []}
        eval_list = []
        ans, gt = {}, {}
        print(len(preds))
        for i, res in enumerate(preds):
            sample_id = res['sample_id']
            gt_ans = self.process(res["gt_response"])
            pred_ans = self.process(res["pred_response"])
            assert gt_ans != ''
            if pred_ans == '':
                s = 0
            else:
                s = rouge.get_scores(pred_ans, gt_ans)[0]['rouge-l']['f']
            acc['f'].append(s)
            eval_list.append({'id':sample_id, 'rouge': s})
            ans[res['sample_id']] = [pred_ans]
            gt[res['sample_id']] = [gt_ans]
        cider_score, cider_scores = cider.compute_score(gt, ans)
        blue_score, blue_scores = bleu.compute_score(gt, ans)

        for i, p in enumerate(preds):
            for j in eval_list:
                if j['id']==p['sample_id']:
                    j['cider'] = cider_scores[i]
                    j['bleu4'] = blue_scores[-1][i]
                
        results = {'rouge': round(np.mean(acc['f'])*100,2), 'cider': round(cider_score*100,2), 'bleu4': round(blue_score[-1]*100,2)}
        return results,eval_list

parser = argparse.ArgumentParser()
parser.add_argument('--result-dir', type=str, default='result')
args = parser.parse_args()
result_dir = args.result_dir
model_name = result_dir.split('/')[-1]

datasets = ['AESOP', 'VIST', 'DM800K', 'Conceptual', 'Animal', 'Vehicle']

for dataset in datasets:
    E = Eval()
    output_dir = os.path.join(result_dir, dataset)
    if not os.path.exists(os.path.join(output_dir,'pred.json')):
        print('%s--%s  No prediction file found'%(model_name, dataset))
        continue
    preds = json.load(open(os.path.join(output_dir,'pred.json'),'r'))
    
    eval_result,eval_list = E.evaluate_rouge(preds)

    print(model_name,end = ':  ')
    print(dataset,end = ':  ')
    print(eval_result)
    with open(os.path.join(output_dir,'eval.json'),'w') as f:
        json.dump(eval_result,f)

    with open(os.path.join(output_dir,'eval_score.json'),'w') as f:
        json.dump(eval_list,f,indent=4)