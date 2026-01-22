from typing import List, Optional
import fire
import json
import time
import os
from tqdm import tqdm
from openai import OpenAI

# 初始化OpenAI客戶端
client = OpenAI(
    api_key=os.getenv("")  # 請確保設置了環境變量
)

class ChatGPTProcessor:
    def __init__(self):
        self.client = client
    
    def get_chatgpt_response(self, prompt, system_prompt, model="gpt-5.2-2025-12-11"):
        """
        獲取ChatGPT回應，使用指定的hyperparameters
        
        Args:
            prompt (str): 用戶提示詞
            system_prompt (str): 系統提示詞
            hyperparams (dict): 超參數配置
            model (str): 使用的模型
            
        Returns:
            str: ChatGPT的回應
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(20)  # 處理rate limit
            return self.get_chatgpt_response(prompt, system_prompt, model)

def generate_(
    data,
    processor,
    max_batch_size,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: Optional[int] = 150):
    
    # 系統提示詞
    system_prompt = ("Use two extremely short sentences to reply. "
                    "The first one is '1. The user prefers xxx .'"
                    "The second one is '2. The item's attributes are xxx.'")

    for i in tqdm(range(int(len(data)/max_batch_size)+1), mininterval=2, desc='  - (Generating)   ', leave=False):
        batch_data = data[i*max_batch_size:(i+1)*max_batch_size]
        if not batch_data:  # 如果batch為空，跳過
            continue
            
        for j, sample in enumerate(batch_data):
            # 構建用戶提示詞
            user_prompt = ("A user bought an item and said \"{explanation}\".  "
                          "Use two sentences to explain the user's preference and the item's attributions, respectively. ".format(
                explanation=sample['explanation']))
            
            # 獲取ChatGPT回應
            result_content = processor.get_chatgpt_response(
                user_prompt, 
                system_prompt
            )
            
            # 解析回應
            try:
                strs = result_content.split('\n')
                # 過濾空行
                strs = [s.strip() for s in strs if s.strip()]
                
                if len(strs) >= 2:
                    # 標準情況：兩行分別包含user preference和item attribution
                    user_pref_line = strs[0]
                    item_attr_line = strs[1]
                    
                    # 提取內容（去除序號）
                    if user_pref_line.startswith('1. '):
                        data[i * max_batch_size + j]['user_preference'] = user_pref_line[3:]
                    else:
                        data[i * max_batch_size + j]['user_preference'] = user_pref_line
                    
                    if item_attr_line.startswith('2. '):
                        data[i * max_batch_size + j]['item_attribution'] = item_attr_line[3:]
                    else:
                        data[i * max_batch_size + j]['item_attribution'] = item_attr_line
                else:
                    # 備用處理：如果格式不符合預期
                    raise Exception("Unexpected format")
                    
            except Exception as e:
                try:
                    # 嘗試其他解析方式
                    if len(strs) >= 1 and '1. ' in strs[0]:
                        # 處理單行包含兩個信息的情況
                        content = strs[0]
                        if ', and ' in content:
                            parts = content.split(', and ')
                            if len(parts) >= 2:
                                data[i * max_batch_size + j]['user_preference'] = parts[0][3:] if parts[0].startswith('1. ') else parts[0]
                                data[i * max_batch_size + j]['item_attribution'] = parts[1]
                                continue
                        
                        # 處理括號內包含第二個信息的情況
                        if '(2. ' in content:
                            idx = content.index('(2. ')
                            user_part = content[:idx].strip()
                            item_part = content[idx+4:].strip().rstrip(')')
                            data[i * max_batch_size + j]['user_preference'] = user_part[3:] if user_part.startswith('1. ') else user_part
                            data[i * max_batch_size + j]['item_attribution'] = item_part
                            continue
                    
                    # 如果所有解析都失敗，使用原始explanation
                    print(f"Failed to parse response: {result_content}")
                    data[i * max_batch_size + j]['user_preference'] = sample['explanation']
                    data[i * max_batch_size + j]['item_attribution'] = sample['explanation']
                    
                except Exception as e2:
                    # 最終備用方案
                    data[i * max_batch_size + j]['user_preference'] = sample['explanation']
                    data[i * max_batch_size + j]['item_attribution'] = sample['explanation']
            
            # 添加小延遲避免API rate limit
            time.sleep(0.1)
    
    return data

def main(
    combat_id: int = 1,
    max_batch_size: int = 10,  # 減少batch size避免API限制
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 150):
    
    # 初始化ChatGPT處理器
    processor = ChatGPTProcessor()
    
    # 讀取數據文件
    data_path = f'./combat_datasets/combat_{combat_id:02d}/explanation.json'
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file {data_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {data_path}")
        return
    
    # 確保數據包含train key
    if 'train' not in data:
        print("Error: 'train' key not found in data")
        return
    
    training_data = data['train']
    
    print(f"Processing combat_{combat_id:02d} with {len(training_data)} samples...")
    
    # 處理訓練數據
    new_training_data = generate_(
        training_data,
        processor,
        max_batch_size,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    # 構建新的數據結構
    new_data = {
        'train': new_training_data
    }
    
    # 保存結果
    output_path = f'./combat_datasets/combat_{combat_id:02d}/explanation_rationale.json'
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(new_data, file, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

def process_all_combats(
    start_combat: int = 1,
    end_combat: int = 100,
    max_batch_size: int = 10,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 150):
    """
    處理所有combat數據集
    
    Args:
        start_combat (int): 開始的combat編號
        end_combat (int): 結束的combat編號
        max_batch_size (int): 批次大小
        temperature (float): 生成溫度
        top_p (float): top_p參數
        max_tokens (int): 最大token數
    """
    
    for combat_id in range(start_combat, end_combat + 1):
        print(f"\n=== Processing Combat {combat_id:02d} ===")
        
        # 檢查文件是否存在
        data_path = f'./combat_datasets/combat_{combat_id:02d}/explanation.json'
        if not os.path.exists(data_path):
            print(f"Skipping combat_{combat_id:02d}: file not found")
            continue
        
        # 處理單個combat
        main(
            combat_id=combat_id,
            max_batch_size=max_batch_size,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        # 在處理完每個combat後稍作休息
        time.sleep(2)
    
    print("\n=== All combats processed! ===")

if __name__ == "__main__":
    fire.Fire({
        'single': main,
        'all': process_all_combats
    })
