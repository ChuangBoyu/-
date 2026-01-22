import os  
import json  
import math  
import random  
import torch  
import datetime  
import glob  
  
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
        combat_dirs.sort()  
          
        print(f"Found {len(combat_dirs)} combat directories")  
          
        # 加載第一個combat資料夾的datamaps.json作為參考  
        self.id2user = {}  
        self.id2item = {}  
          
        for combat_dir in combat_dirs:  
            exp_file = os.path.join(combat_dir, 'explanation_rationale.json')  
            datamaps_file = os.path.join(combat_dir, 'datamaps.json')  
              
            if os.path.exists(exp_file) and os.path.exists(datamaps_file):  
                # 加載datamaps  
                with open(datamaps_file, 'r', encoding='utf-8') as f:  
                    datamaps = json.load(f)  
                    if not self.id2user:  # 只在第一次設置用戶映射  
                        self.id2user = datamaps['id2user']  
                    # 合併item映射  
                    self.id2item.update(datamaps['id2item'])  
                  
                # 加載解釋數據  
                with open(exp_file, 'r', encoding='utf-8') as f:  
                    combat_data = json.load(f)  
                    all_train_data.extend(combat_data.get('train', []))  
                    all_val_data.extend(combat_data.get('val', []))  
                    all_test_data.extend(combat_data.get('test', []))  
          
        self.train = all_train_data  
        self.valid = all_val_data  
        self.test = all_test_data  
          
        print(f"Loaded {len(self.train)} training samples, {len(self.valid)} validation samples, {len(self.test)} test samples")  
  
def compute_whole_word_id(seq_batch, tokenizer, max_len):  
    whole_word_ids = []  
    for seq in seq_batch:  
        token_list = tokenizer.tokenize(seq)  
        start_indices = []  
        for idx, token in enumerate(token_list):  
            if token == '_' or token == ':':  
                start_indices.append(idx - 1)  
        end_indices = []  
        for start in start_indices:  
            mover = start + 2  
            while mover < len(token_list) and (token_list[mover].isdigit() or token_list[mover].isalpha()):  
                mover += 1  
            end_indices.append(mover)  
        whole_word_id = [0] * len(token_list)  
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):  
            if start >= 0 and end <= len(token_list):  
                whole_word_id[start:end] = [i + 1] * (end - start)  
        whole_word_ids.append(whole_word_id)  
  
    padded_whole_word_ids = []  
    for whole_word_id in whole_word_ids:  
        padded_whole_word_ids.append(whole_word_id + [0] * (max_len - len(whole_word_id)))  
  
    return padded_whole_word_ids  
  
class ExpBatchify:  
    def __init__(self, exp_data, tokenizer, exp_len, batch_size, id2item=None):  
        self.task_id = 0  
        self.id2item = id2item or {}  
        input_list, output_list = [], []  
          
        for x in exp_data:  
            # 方案1：如果有實際戰鬥內容，使用它；否則使用ID  
            if self.id2item and str(x['item']) in self.id2item:  
                combat_content = self.id2item[str(x['item'])]  
                template = 'user_{} combat:{}' 
                input_list.append(template.format(x['user'], combat_content))  
            else:  
                # 回退到原始模板  
                template = 'user_{} item_{}'  
                input_list.append(template.format(x['user'], x['item']))  
            output_list.append(x['explanation'])  
  
        encoded_source = tokenizer(input_list, padding=True, return_tensors='pt', truncation=True, max_length=512)  
        self.source_seq = encoded_source['input_ids'].contiguous()  
        self.source_mask = encoded_source['attention_mask'].contiguous()  
        max_len = self.source_seq.size(1)  
        whole_word_ids = compute_whole_word_id(input_list, tokenizer, max_len)  
        self.whole_word = torch.tensor(whole_word_ids, dtype=torch.int64).contiguous()  
        encoded_target = tokenizer(output_list, padding=True, return_tensors='pt', truncation=True, max_length=exp_len)  
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
    def __init__(self, model, tokenizer, device, exp_len=200, debug=False):  
        self.model = model  
        self.tokenizer = tokenizer  
        self.device = device  
        self.exp_len = exp_len  
        self.task_id = 0  
        self.debug = debug
          
    def generate_explanation(self, user_id, item_id):  
        """原始方法：使用ID生成解釋"""  
        template = 'user_{} item_{}'  
        input_text = template.format(user_id, item_id)  
        return self._generate_from_text(input_text)  
          
    def generate_explanation_with_content(self, user_id, combat_flow_text):  
        """使用實際戰鬥內容生成解釋"""  
        template = 'user_{} analyzing detailed combat: {}'  
        input_text = template.format(user_id, combat_flow_text)  
          
        if self.debug:  
            print(f"Input template: {template}")  
            print(f"Full input text: {input_text[:200]}...")  
            print(f"Combat flow hash: {hash(combat_flow_text)}")  
          
        return self._generate_from_text(input_text)   
      
    def _generate_from_text(self, input_text):  
        """共用的生成邏輯"""  
        # 增加輸入長度限制以處理更長的戰鬥序列  
        if len(input_text) > 1000:  # 從 800 增加到 1000  
            input_text = input_text[:1000] + "..." 
          
        if self.debug:  
            print(f"Tokenizing input of length: {len(input_text)}")  
          
        encoded_source = self.tokenizer([input_text], padding=True, return_tensors='pt',   
                                      truncation=True, max_length=1024)  
        source_seq = encoded_source['input_ids'].to(self.device)  
        source_mask = encoded_source['attention_mask'].to(self.device)  
          
        if self.debug:  
            print(f"Token sequence length: {source_seq.size(1)}")  
            print(f"First 10 tokens: {source_seq[0][:10].tolist()}")  
          
        max_len = source_seq.size(1)  
        whole_word_ids = compute_whole_word_id([input_text], self.tokenizer, max_len)  
        whole_word = torch.tensor(whole_word_ids, dtype=torch.int64).to(self.device)  
          
        task = torch.ones((1,), dtype=torch.int64).to(self.device) * self.task_id  
          
        self.model.eval()  
        with torch.no_grad():  
            try:  
                #outputs = self.model.beam_search(  
                #    task_id=task,  
                #    input_ids=source_seq,  
                #    whole_word_ids=whole_word,  
                #    attention_mask=source_mask,  
                #    max_length=min(self.exp_len, 80),  
                #    num_beams=7,  # 增加beam數量以獲得更多樣化的輸出  
                #    early_stopping=True,  
                #    min_length=10,  
                #    num_return_sequences=1,  
                #   diversity_penalty=0.7  # 添加多樣性懲罰  

                outputs = self.model.generate(
                input_ids=source_seq,
                attention_mask=source_mask,
                max_length=min(self.exp_len,120),
                min_length=10,
                # --- 核心修改 ---
                do_sample=True,          # 啟用採樣
                temperature=1,        # 設定溫度，增加多樣性
                top_k=50,                # 設定 top_k
                top_p=0.95,               # 設定 top_p
                # ------------------
                num_return_sequences=1,  # 仍然只返回一個最佳序列
                # 您原有的其他參數可以視情況保留或移除
                task_id=task,
                whole_word_ids=whole_word,   
                )  
                  
                if outputs.numel() == 0:  
                    return "Unable to generate commentary for this combat sequence."  
                      
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
                  
                if self.debug:  
                    print(f"Generated text: {generated_text}")  
                  
                return generated_text  
                  
            except Exception as e:  
                print(f"Generation failed: {e}")  
                return f"Error generating commentary: {str(e)}"
  
def ids2tokens(ids, tokenizer):  
    text = tokenizer.decode(ids, skip_special_tokens=True)  
    return text.split()