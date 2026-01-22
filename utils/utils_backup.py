import os  
import json  
import math  
import random  
import torch  
import datetime  
import glob  
from utils.rouge import rouge  
from utils.bleu import compute_bleu  
  
def now_time():  
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '  
  
def rouge_score(references, generated):  
    """both are a list of strings"""  
    score = rouge(generated, references)  
    rouge_s = {k: (v * 100) for (k, v) in score.items()}  
    return rouge_s  
  
def bleu_score(references, generated, n_gram=4, smooth=False):  
    """a list of lists of tokens"""  
    formatted_ref = [[ref] for ref in references]  
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)  
    return bleu_s * 100  
  
class ExpDataLoader:  
    def __init__(self, data_dir):  
        # 合併所有combat資料夾的數據  
        all_train_data = []  
        all_val_data = []  
        all_test_data = []  
          
        # 尋找所有combat_xx資料夾  
        combat_dirs = glob.glob(os.path.join(data_dir, 'combat_*'))  
        combat_dirs.sort()  # 確保順序一致  
          
        print(f"Found {len(combat_dirs)} combat directories")  
          
        for combat_dir in combat_dirs:  
            exp_file = os.path.join(combat_dir, 'explanation_rationale.json')  
            if os.path.exists(exp_file):  
                with open(exp_file, 'r', encoding='utf-8') as f:  
                    combat_data = json.load(f)  
                      
                all_train_data.extend(combat_data.get('train', []))  
                all_val_data.extend(combat_data.get('val', []))  
                all_test_data.extend(combat_data.get('test', []))  
          
        self.train = all_train_data  
        self.valid = all_val_data  
        self.test = all_test_data  
          
        print(f"Loaded {len(self.train)} training samples, {len(self.valid)} validation samples, {len(self.test)} test samples")  
          
        # 加載第一個combat資料夾的datamaps.json作為參考  
        if combat_dirs:  
            datamaps_file = os.path.join(combat_dirs[0], 'datamaps.json')  
            if os.path.exists(datamaps_file):  
                with open(datamaps_file, 'r', encoding='utf-8') as f:  
                    datamaps = json.load(f)  
                self.id2user = datamaps['id2user']  
                self.id2item = datamaps['id2item']  
            else:  
                # 如果沒有datamaps.json，創建默認映射  
                self.id2user = {  
                    "0": "aggressive_commentator",  
                    "1": "defensive_analyst",   
                    "2": "technical_expert",  
                    "3": "entertainment_focused"  
                }  
                self.id2item = {}  
  
def compute_whole_word_id(seq_batch, tokenizer, max_len):  
    whole_word_ids = []  
    for seq in seq_batch:  
        token_list = tokenizer.tokenize(seq)  
        start_indices = []  
        for idx, token in enumerate(token_list):  
            if token == '_':  
                start_indices.append(idx - 1)  
        end_indices = []  
        for start in start_indices:  
            mover = start + 2  
            while mover < len(token_list) and token_list[mover].isdigit():  
                mover += 1  
            end_indices.append(mover)  
        whole_word_id = [0] * len(token_list)  
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):  
            whole_word_id[start:end] = [i + 1] * (end - start)  
        whole_word_ids.append(whole_word_id)  
  
    padded_whole_word_ids = []  
    for whole_word_id in whole_word_ids:  
        padded_whole_word_ids.append(whole_word_id + [0] * (max_len - len(whole_word_id)))  
  
    return padded_whole_word_ids  
  
class ExpBatchify:  
    def __init__(self, exp_data, tokenizer, exp_len, batch_size):  
        self.task_id = 0  
        template = 'user_{} item_{}'  
        input_list, output_list = [], []  
          
        for x in exp_data:  
            input_list.append(template.format(x['user'], x['item']))  
            output_list.append(x['explanation'])  
  
        encoded_source = tokenizer(input_list, padding=True, return_tensors='pt')  
        self.source_seq = encoded_source['input_ids'].contiguous()  
        self.source_mask = encoded_source['attention_mask'].contiguous()  
        max_len = self.source_seq.size(1)  
        whole_word_ids = compute_whole_word_id(input_list, tokenizer, max_len)  
        self.whole_word = torch.tensor(whole_word_ids, dtype=torch.int64).contiguous()  
        encoded_target = tokenizer(output_list, padding=True, return_tensors='pt')  
        self.target_seq = encoded_target['input_ids'][:, :exp_len].contiguous()  
        self.batch_size = batch_size  
        self.sample_num = len(exp_data)  
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))  
        self.step = 0  
  
    def next_batch(self):  
        if self.step == self.total_step:  
            self.step = 0  
  
        start = self.step * self.batch_size  
        offset = min(start + self.batch_size, self.sample_num)  
        self.step += 1  
        source_seq = self.source_seq[start:offset]  
        source_mask = self.source_mask[start:offset]  
        whole_word = self.whole_word[start:offset]  
        target_seq = self.target_seq[start:offset]  
        task = torch.ones((offset - start,), dtype=torch.int64) * self.task_id  
        return task, source_seq, source_mask, whole_word, target_seq  
  
    def next_batch_valid(self):  
        return self.next_batch()  
  
    def next_batch_test(self):  
        return self.next_batch()  
  
class DynamicExpGenerator:  
    def __init__(self, model, tokenizer, device, exp_len=100):  
        self.model = model  
        self.tokenizer = tokenizer  
        self.device = device  
        self.exp_len = exp_len  
        self.task_id = 0  
          
    def generate_explanation(self, user_id, item_id):  
        template = 'user_{} item_{}'  
        input_text = template.format(user_id, item_id)  
          
        encoded_source = self.tokenizer([input_text], padding=True, return_tensors='pt')  
        source_seq = encoded_source['input_ids'].to(self.device)  
        source_mask = encoded_source['attention_mask'].to(self.device)  
          
        max_len = source_seq.size(1)  
        whole_word_ids = compute_whole_word_id([input_text], self.tokenizer, max_len)  
        whole_word = torch.tensor(whole_word_ids, dtype=torch.int64).to(self.device)  
          
        task = torch.ones((1,), dtype=torch.int64).to(self.device) * self.task_id  
          
        self.model.eval()  
        with torch.no_grad():  
            outputs = self.model.generate(  
                task=task,  
                input_ids=source_seq,  
                whole_word_ids=whole_word,  
                attention_mask=source_mask,  
                max_length=self.exp_len,  
                num_beams=5,  
                early_stopping=True,  
                do_sample=False  
            )  
              
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
        return generated_text  
  
def ids2tokens(ids, tokenizer):  
    text = tokenizer.decode(ids, skip_special_tokens=True)  
    return text.split()