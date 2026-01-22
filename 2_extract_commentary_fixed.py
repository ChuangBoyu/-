import json
import re
import os

def extract_commentary_data():
    """
    從enhanced_style_comparison.txt提取資料並生成結構化資料夾和JSON檔案
    """
    
    # 讀取文件
    try:
        with open('generated_commentaries/enhanced_style_comparison.txt', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("找不到文件: generated_commentaries/enhanced_style_comparison.txt")
        return
    
    # 用戶ID映射
    id2user = {
        "0": "aggressive_commentator",
        "1": "defensive_analyst", 
        "2": "technical_expert",
        "3": "entertainment_focused"
    }
    
    # 創建主資料夾
    main_folder = "combat_datasets"
    os.makedirs(main_folder, exist_ok=True)
    
    # 按COMBAT DIRECTORY分割內容
    combat_sections = re.split(r'【COMBAT DIRECTORY: (\d+)】', content)[1:]  # 跳過第一個空白部分
    
    # 處理每個combat section
    for i in range(0, len(combat_sections), 2):
        combat_id = combat_sections[i].strip()
        combat_content = combat_sections[i+1].strip()
        
        print(f"處理 COMBAT DIRECTORY: {combat_id}")
        
        # 創建單個combat資料夾
        combat_folder = os.path.join(main_folder, f"combat_{combat_id.zfill(2)}")
        os.makedirs(combat_folder, exist_ok=True)
        
        # 分割每個frame data block
        frame_blocks = re.split(r'Frame Data:', combat_content)[1:]  # 跳過第一個空白部分
        
        train_data = []
        id2item = {}
        item_counter = 0
        
        for block in frame_blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            frame_info = lines[0].strip()
            
            # 保留完整的Frame Data信息作為item描述
            id2item[str(item_counter)] = frame_info
            
            # 提取四種評論
            comments = {}
            current_style = None
            
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('AGGRESSIVE:'):
                    current_style = 'AGGRESSIVE'
                    comments[current_style] = line[11:].strip()
                elif line.startswith('DEFENSIVE:'):
                    current_style = 'DEFENSIVE'
                    comments[current_style] = line[10:].strip()
                elif line.startswith('TECHNICAL:'):
                    current_style = 'TECHNICAL'
                    comments[current_style] = line[10:].strip()
                elif line.startswith('ENTERTAINMENT:'):
                    current_style = 'ENTERTAINMENT'
                    comments[current_style] = line[14:].strip()
                elif current_style and line:
                    # 繼續上一個評論
                    comments[current_style] += ' ' + line
            
            # 為每種評論風格創建訓練數據
            style_to_user = {
                'AGGRESSIVE': 0,
                'DEFENSIVE': 1,
                'TECHNICAL': 2,
                'ENTERTAINMENT': 3
            }
            
            for style, comment in comments.items():
                if style in style_to_user:
                    train_data.append({
                        "user": style_to_user[style],
                        "item": item_counter,
                        "explanation": comment
                    })
            
            item_counter += 1
        
        # 生成explanation.json (只包含train資料)
        explanation_data = {
            "train": train_data
        }
        
        explanation_path = os.path.join(combat_folder, "explanation.json")
        with open(explanation_path, 'w', encoding='utf-8') as f:
            json.dump(explanation_data, f, indent=4, ensure_ascii=False)
        
        # 生成datamaps.json (只包含id2user和id2item)
        datamaps_data = {
            "id2user": id2user,
            "id2item": id2item
        }
        
        datamaps_path = os.path.join(combat_folder, "datamaps.json")
        with open(datamaps_path, 'w', encoding='utf-8') as f:
            json.dump(datamaps_data, f, indent=4, ensure_ascii=False)
        
        # 生成sequential.txt
        # 收集每個用戶使用的items
        user_sequences = {}
        for data in train_data:
            user_id = data["user"]
            item_id = data["item"]
            
            if user_id not in user_sequences:
                user_sequences[user_id] = []
            user_sequences[user_id].append(item_id)
        
        # 寫入sequential.txt
        sequential_path = os.path.join(combat_folder, "sequential.txt")
        with open(sequential_path, 'w', encoding='utf-8') as f:
            for user_id in sorted(user_sequences.keys()):
                items = user_sequences[user_id]
                # 格式: user_id item_1 item_2 ... item_n
                line = f"{user_id} " + " ".join(map(str, items)) + "\n"
                f.write(line)
        
        # 生成空的negative.txt (符合研究source code需求)
        negative_path = os.path.join(combat_folder, "negative.txt")
        with open(negative_path, 'w', encoding='utf-8') as f:
            pass  # 創建空檔案
        
        print(f"生成資料夾: {combat_folder}")
        print(f"  - explanation.json (包含 {len(train_data)} 筆訓練資料)")
        print(f"  - datamaps.json (包含 {len(id2item)} 個items)")
        print(f"  - sequential.txt (包含 {len(user_sequences)} 個用戶序列)")
        print(f"  - negative.txt (空檔案)")

def create_full_structure():
    """
    創建完整的100個combat資料夾結構（示例用）
    """
    main_folder = "combat_datasets"
    
    # 如果需要創建100個資料夾的示例結構
    print("\n是否要創建100個combat資料夾的示例結構？(輸入 'yes' 確認)")
    response = input().strip().lower()
    
    if response == 'yes':
        for i in range(1, 101):
            combat_folder = os.path.join(main_folder, f"combat_{i:02d}")
            os.makedirs(combat_folder, exist_ok=True)
            
            # 創建空的JSON檔案作為示例
            if not os.path.exists(os.path.join(combat_folder, "explanation.json")):
                example_explanation = {
                    "train": [
                        {
                            "user": 0,
                            "item": 0,
                            "explanation": f"Example explanation for combat {i:02d}"
                        }
                    ]
                }
                with open(os.path.join(combat_folder, "explanation.json"), 'w', encoding='utf-8') as f:
                    json.dump(example_explanation, f, indent=4, ensure_ascii=False)
            
            if not os.path.exists(os.path.join(combat_folder, "datamaps.json")):
                example_datamaps = {
                    "id2user": {
                        "0": "aggressive_commentator",
                        "1": "defensive_analyst",
                        "2": "technical_expert",
                        "3": "entertainment_focused"
                    },
                    "id2item": {
                        "0": f"Frame 1: Self changed from STAND to CROUCH → Frame 5: Self took 10 damage → Frame 15: Self changed from CROUCH to STAND (Example for combat {i:02d})"
                    }
                }
                with open(os.path.join(combat_folder, "datamaps.json"), 'w', encoding='utf-8') as f:
                    json.dump(example_datamaps, f, indent=4, ensure_ascii=False)
            
            # 創建示例sequential.txt
            if not os.path.exists(os.path.join(combat_folder, "sequential.txt")):
                with open(os.path.join(combat_folder, "sequential.txt"), 'w', encoding='utf-8') as f:
                    f.write(f"0 0\n")  # user 0 使用 item 0
            
            # 創建空的negative.txt
            if not os.path.exists(os.path.join(combat_folder, "negative.txt")):
                with open(os.path.join(combat_folder, "negative.txt"), 'w', encoding='utf-8') as f:
                    pass  # 創建空檔案

        
        print(f"已創建100個combat資料夾結構在 {main_folder} 中")

if __name__ == "__main__":
    extract_commentary_data()
    print("資料提取完成！")
    
    create_full_structure()
    
    print(f"\n資料夾結構:")
    print(f"combat_datasets/")
    print(f"├── combat_01/")
    print(f"│   ├── explanation.json")
    print(f"│   ├── datamaps.json")
    print(f"│   ├── sequential.txt")
    print(f"│   └── negative.txt")
    print(f"├── combat_02/")
    print(f"│   ├── explanation.json")
    print(f"│   ├── datamaps.json")
    print(f"│   ├── sequential.txt")
    print(f"│   └── negative.txt")
    print(f"└── ...")
