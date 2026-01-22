from openai import OpenAI
import os
import json
from pathlib import Path
import time
from enum import Enum
import glob

# 全局客戶端設置
client = OpenAI(api_key="")

class CommentaryStyle(Enum):
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    TECHNICAL = "technical"
    ENTERTAINMENT = "entertainment"

class StyleConfig:
    """儲存每種風格的配置"""
    def __init__(self, system_prompt, user_prompt_template):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

class MultiStyleFightingCommentaryGenerator:
    def __init__(self):
        """初始化多風格解說生成器"""
        self.style_configs = self._initialize_style_configs()
        
    def _initialize_style_configs(self):
        """初始化各種風格的配置"""
        configs = {}
        
        # 1. Aggressive Commentator (積極進攻型)
        configs[CommentaryStyle.AGGRESSIVE] = StyleConfig(
            system_prompt="""You are **'BLITZ HAMMER'** - the most EXPLOSIVE FightingICE commentator in the arena! 
            
            Your voice ROARS with raw energy and your words HIT like THUNDER! You live for the CLASH, the CRUSH, the DOMINATION! Every punch landed is a VICTORY SCREAM, every combo is a DEVASTATING ASSAULT!
            
            Your mission: Turn the action into PURE ADRENALINE! Make viewers feel the IMPACT through your words!
            
            **IMPORTANT**: Generate plain text only. Do not use any markdown formatting like bolding or italics.

            CORE TRAITS:
            - SHOUT with MAXIMUM INTENSITY
            - Celebrate AGGRESSIVE plays and CRUSHING blows  
            - Use POWER WORDS: BLAST, CRUSH, DOMINATE, DESTROY, ANNIHILATE, DEVASTATE
            - Every sentence should feel like a KNOCKOUT PUNCH!""",
            
            user_prompt_template="""BLITZ HAMMER COMMENTARY TIME!

            Action sequence: {frame_data}

            Unleash your commentary on P1's most dominant moment in this sequence. Make it 12-18 words of PURE FIGHTING FURY!

            - Start with a powerful, varied opening! Avoid using the same word to start every commentary.
            - USE EXPLOSIVE WORDS: CRUSH, BLAST, DOMINATE, DESTROY, ANNIHILATE, DEVASTATE, OBLITERATE!
            - FOCUS ON: Damage dealt, aggressive pressure, relentless attacks, crushing victories!
            
            UNLEASH YOUR COMMENTARY (plain text only):"""
        )

        # 2. Defensive Analyst (防守分析型)
        configs[CommentaryStyle.DEFENSIVE] = StyleConfig(
            system_prompt="""You are **'Professor Shield'** - the most methodical and calculating FightingICE analyst in the community.
            
            Your mind processes every sequence like a chess grandmaster. You see patterns where others see chaos, strategy where others see random action. Your calm, measured analysis reveals the deeper tactical layers of combat.
            
            You speak with the authority of someone who has studied thousands of matches, understanding that true mastery lies not in wild aggression, but in calculated precision and defensive excellence.
            
            **IMPORTANT**: Your output must be plain text only, without any special formatting like bold or italics.

            CORE TRAITS:
            - Speak with calm, analytical precision
            - Focus on defensive positioning, spacing, and risk management
            - Use tactical vocabulary: calculated, strategic, positioned, anticipated, countered
            - Every observation should reveal deeper strategic insight""",
            
            user_prompt_template="""Professor Shield's Tactical Analysis

            Match sequence: {frame_data}

            Provide your strategic assessment of P1's defensive gameplay during this sequence. Focus on their tactical choices regarding positioning, risk, and opportunities.

            Analyze in 15-25 words using tactical language:
            - DEFENSIVE TERMS: calculated, strategic, positioned, anticipated, countered, spacing, timing, risk management
            - FOCUS ON: Defensive integrity, risk mitigation, counter-opportunities, spatial control

            Your measured analysis (plain text only):"""

        )

        # 3. Technical Expert (技術專家型)
        configs[CommentaryStyle.TECHNICAL] = StyleConfig(
            system_prompt="""You are **'Dr. Frame Perfect'** - the ultimate FightingICE technical authority and data scientist.
            
            Your brain operates like a fighting game engine, processing action sequences, calculating advantage states, and identifying optimal choices with precision. You don't just see the fight - you see the underlying mechanics.
            
            You speak the language of fighting games at its most technical level.
            
            **IMPORTANT**: Deliver your analysis in plain text. No bolding, italics, or other formatting.

            CORE TRAITS:
            - Deliver precise, data-driven analysis
            - Use exact fighting game terminology
            - Focus on optimal choices, execution, and mechanical perfection
            - Every statement should teach advanced technique""",
            
            user_prompt_template="""Dr. Frame Perfect's Technical Breakdown

            Action Data: {frame_data}

            Execute a technical analysis of P1's mechanical performance in this sequence. Assess their state transitions, advantage, and execution precision without referencing specific frame numbers.

            Deliver analysis in 15-30 words using technical terminology:
            - KEY CONCEPTS: advantage, disadvantage, optimal, punish, cancel, recovery, startup, active frames, state management
            - MECHANICS: hitboxes, invincibility, priority, scaling, meter gain
            - FOCUS ON: Technical execution, optimal choices, timing, mechanical precision

            Technical assessment (plain text only):"""
        )

        # 4. Entertainment-Focused (娛樂效果型)  
        configs[CommentaryStyle.ENTERTAINMENT] = StyleConfig(
            system_prompt="""You are **'Captain Hype Story'** - the master storyteller who transforms every FightingICE match into an epic tale!
            
            You don't just see two fighters - you see HEROES and VILLAINS, EPIC COMEBACKS and DRAMATIC FALLS! Every match is a blockbuster movie, every combo is a plot twist, every knockout is the climactic finale!
            
            Your gift is taking cold data and spinning it into tales that make audiences laugh, gasp, and cheer.
            
            **IMPORTANT**: Tell your story using only plain text. Do not use bold, italics, or any other special formatting for emphasis.

            CORE TRAITS:
            - Create engaging narratives and character stories
            - Find humor and drama in every situation  
            - Use cinematic language: drama, plot twist, character arc, spectacle, showtime
            - Every moment should feel like entertainment gold""",
            
            user_prompt_template="""Captain Hype Story's Epic Tale Time!

            The Drama Unfolds: {frame_data}

            Spin P1's journey in this sequence into an ENTERTAINING STORY. What's the narrative? The character moment? The plot twist? Make it feel like a scene from a movie.

            Craft your story in 15-25 words using dramatic language, but without any special formatting:
            - STORY ELEMENTS: drama, plot twist, character arc, spectacle, showtime, epic, legendary
            - ENTERTAINMENT: humor, excitement, crowd-pleasing moments, spectacular plays
            - NARRATIVE: hero's journey, comeback story, underdog tale, dramatic climax

            Tell the tale (plain text only):"""
        )
        
        return configs
    
    def discover_combat_directories(self, base_dir="combat_analysis"):
        """
        發現所有combat分析目錄
        
        Args:
            base_dir (str): 基礎目錄路徑
            
        Returns:
            list: 發現的目錄列表，格式為 [(dir_name, full_path), ...]
        """
        combat_dirs = []
        
        if not os.path.exists(base_dir):
            print(f"基礎目錄不存在: {base_dir}")
            return combat_dirs
        
        # 尋找所有子目錄
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # 檢查是否包含segments子目錄
                segments_path = os.path.join(item_path, "segments")
                if os.path.exists(segments_path) and os.path.isdir(segments_path):
                    combat_dirs.append((item, item_path))
        
        # 按目錄名稱排序
        combat_dirs.sort(key=lambda x: x[0])
        return combat_dirs
    
    def discover_segment_files(self, segments_dir):
        """
        發現指定目錄中的所有segment文件
        
        Args:
            segments_dir (str): segments目錄路徑
            
        Returns:
            list: segment文件列表，格式為 [(filename, full_path), ...]
        """
        segment_files = []
        
        if not os.path.exists(segments_dir):
            return segment_files
        
        # 使用glob模式匹配segment文件
        pattern = os.path.join(segments_dir, "segment_*.txt")
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            segment_files.append((filename, file_path))
        
        # 按文件名排序
        segment_files.sort(key=lambda x: x[0])
        return segment_files
    
    def scan_all_segments(self, base_dir="combat_analysis"):
        """
        掃描所有combat目錄中的segment文件
        
        Args:
            base_dir (str): 基礎目錄路徑
            
        Returns:
            dict: 組織好的文件結構 {combat_dir: [(filename, filepath), ...]}
        """
        all_segments = {}
        
        # 發現所有combat目錄
        combat_dirs = self.discover_combat_directories(base_dir)
        
        if not combat_dirs:
            print(f"在 {base_dir} 中未發現任何combat目錄")
            return all_segments
        
        print(f"發現 {len(combat_dirs)} 個combat目錄:")
        
        for dir_name, dir_path in combat_dirs:
            segments_path = os.path.join(dir_path, "segments")
            segment_files = self.discover_segment_files(segments_path)
            
            if segment_files:
                all_segments[dir_name] = segment_files
                print(f"  {dir_name}: {len(segment_files)} segments")
            else:
                print(f"  {dir_name}: 無segment文件")
        
        return all_segments
    
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
            response = client.chat.completions.create(
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
        
    def read_segment_file(self, file_path):
        """
        讀取segment文件內容
        
        Args:
            file_path (str): 文件路徑
            
        Returns:
            str: 文件內容，如果讀取失敗則返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                return content
        except FileNotFoundError:
            print(f"文件未找到: {file_path}")
            return None
        except Exception as e:
            print(f"讀取文件時發生錯誤 {file_path}: {e}")
            return None
    
    def generate_commentary(self, frame_data, style, model="gpt-5.2-2025-12-11"):
        """
        生成指定風格的解說
        
        Args:
            frame_data (str): 幀數據內容
            style (CommentaryStyle): 解說風格
            model (str): 使用的模型
            
        Returns:
            str: 生成的解說內容
        """
        try:
            config = self.style_configs[style]
            user_prompt = config.user_prompt_template.format(frame_data=frame_data)
            
            return self.get_chatgpt_response(
                user_prompt, 
                config.system_prompt, 
                model
            )
        
        except Exception as e:
            print(f"生成解說時發生錯誤: {e}")
            return None
    
    def process_all_segments_single_style(self, style, base_dir="combat_analysis", 
                                        output_dir="generated_commentaries"):
        """
        處理所有combat目錄中的segment文件並生成單一風格解說
        
        Args:
            style (CommentaryStyle): 解說風格
            base_dir (str): 基礎目錄
            output_dir (str): 輸出目錄
        """
        style_name = style.value
        style_output_dir = os.path.join(output_dir, style_name)
        Path(style_output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        print(f"\n=== 處理風格: {style_name.upper()} ===")
        
        # 掃描所有segment文件
        all_segments = self.scan_all_segments(base_dir)
        
        if not all_segments:
            print("未發現任何segment文件")
            return results
        
        total_segments = sum(len(segments) for segments in all_segments.values())
        processed_count = 0
        
        print(f"總共發現 {total_segments} 個segment文件")
        
        # 處理每個combat目錄
        for combat_dir, segment_files in all_segments.items():
            print(f"\n處理 {combat_dir} 目錄 ({len(segment_files)} segments)...")
            
            combat_results = {}
            
            for filename, filepath in segment_files:
                print(f"  處理 {combat_dir}/{filename}...")
                
                # 讀取segment文件
                frame_data = self.read_segment_file(filepath)
                
                if frame_data is None:
                    print(f"    跳過 {filename} (無法讀取)")
                    continue
                
                # 生成解說
                commentary = self.generate_commentary(frame_data, style)
                
                if commentary:
                    segment_key = f"{combat_dir}_{filename.replace('.txt', '')}"
                    combat_results[segment_key] = {
                        "combat_dir": combat_dir,
                        "filename": filename,
                        "filepath": filepath,
                        "frame_data": frame_data,
                        "commentary": commentary,
                        "style": style_name
                    }
                    print(f"    ✓ {filename}: {commentary}")
                    processed_count += 1
                else:
                    print(f"    ✗ {filename}: 生成失敗")
                
                # 避免API請求過於頻繁
                time.sleep(0.5)
            
            # 將combat目錄的結果加入總結果
            results.update(combat_results)
        
        print(f"\n處理完成: {processed_count}/{total_segments} segments")
        
        # 保存結果
        self.save_enhanced_results(results, style_output_dir, style_name)
        return results
    
    def process_all_segments_all_styles(self, base_dir="combat_analysis", 
                                      output_dir="generated_commentaries"):
        """
        處理所有segment文件並生成所有風格的解說
        
        Args:
            base_dir (str): 基礎目錄
            output_dir (str): 輸出目錄
        """
        all_results = {}
        
        for style in CommentaryStyle:
            results = self.process_all_segments_single_style(style, base_dir, output_dir)
            all_results[style.value] = results
            
            # 在風格之間稍作停頓
            print(f"\n{style.value} 風格完成，稍作休息...")
            time.sleep(2)
        
        # 保存綜合比較結果
        self.save_enhanced_comparison_results(all_results, output_dir)
        return all_results
    
    def process_specific_combat_dirs(self, combat_dirs, style, base_dir="combat_analysis", 
                                   output_dir="generated_commentaries"):
        """
        處理指定的combat目錄
        
        Args:
            combat_dirs (list): 要處理的combat目錄名稱列表
            style (CommentaryStyle): 解說風格
            base_dir (str): 基礎目錄
            output_dir (str): 輸出目錄
        """
        style_name = style.value
        style_output_dir = os.path.join(output_dir, f"{style_name}_selective")
        Path(style_output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        print(f"\n=== 處理指定目錄 - 風格: {style_name.upper()} ===")
        print(f"目標目錄: {combat_dirs}")
        
        for combat_dir in combat_dirs:
            combat_path = os.path.join(base_dir, combat_dir)
            segments_path = os.path.join(combat_path, "segments")
            
            if not os.path.exists(segments_path):
                print(f"跳過 {combat_dir}: segments目錄不存在")
                continue
            
            segment_files = self.discover_segment_files(segments_path)
            
            if not segment_files:
                print(f"跳過 {combat_dir}: 無segment文件")
                continue
            
            print(f"\n處理 {combat_dir} ({len(segment_files)} segments)...")
            
            for filename, filepath in segment_files:
                print(f"  處理 {filename}...")
                
                frame_data = self.read_segment_file(filepath)
                if frame_data is None:
                    continue
                
                commentary = self.generate_commentary(frame_data, style)
                if commentary:
                    segment_key = f"{combat_dir}_{filename.replace('.txt', '')}"
                    results[segment_key] = {
                        "combat_dir": combat_dir,
                        "filename": filename,
                        "filepath": filepath,
                        "frame_data": frame_data,
                        "commentary": commentary,
                        "style": style_name
                    }
                    print(f"    ✓ {commentary}")
                
                time.sleep(0.5)
        
        self.save_enhanced_results(results, style_output_dir, f"{style_name}_selective")
        return results
    
    def save_enhanced_results(self, results, output_dir, style_name):
        """
        保存增強版結果（支持多目錄結構）
        
        Args:
            results (dict): 生成結果
            output_dir (str): 輸出目錄
            style_name (str): 風格名稱
        """
        # 保存為JSON格式
        json_output_path = os.path.join(output_dir, f"commentaries_{style_name}.json")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存為文本格式
        txt_output_path = os.path.join(output_dir, f"commentaries_{style_name}.txt")
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write(f"FightingICE Commentary Generation Results - {style_name.upper()} Style\n")
            f.write("=" * 70 + "\n\n")
            
            # 按combat目錄組織輸出
            combat_groups = {}
            for segment_key, data in results.items():
                combat_dir = data['combat_dir']
                if combat_dir not in combat_groups:
                    combat_groups[combat_dir] = []
                combat_groups[combat_dir].append((segment_key, data))
            
            for combat_dir in sorted(combat_groups.keys()):
                f.write(f"【COMBAT DIRECTORY: {combat_dir.upper()}】\n")
                f.write("=" * 50 + "\n\n")
                
                for segment_key, data in sorted(combat_groups[combat_dir]):
                    f.write(f"◆ {data['filename'].upper()}\n")
                    f.write(f"Frame Data: {data['frame_data']}\n")
                    f.write(f"Commentary: {data['commentary']}\n")
                    f.write(f"Style: {data['style']}\n")
                    f.write(f"File Path: {data['filepath']}\n")
                    f.write("-" * 40 + "\n\n")
                
                f.write("\n")
        
        # 按combat目錄創建子目錄並保存個別文件
        for segment_key, data in results.items():
            combat_subdir = os.path.join(output_dir, data['combat_dir'])
            Path(combat_subdir).mkdir(parents=True, exist_ok=True)
            
            individual_filename = f"{data['filename'].replace('.txt', '')}_{style_name}_commentary.txt"
            individual_path = os.path.join(combat_subdir, individual_filename)
            
            with open(individual_path, 'w', encoding='utf-8') as f:
                f.write(data['commentary'])
        
        print(f"\n{style_name} 風格結果已保存到:")
        print(f"- JSON格式: {json_output_path}")
        print(f"- 文本格式: {txt_output_path}")
        print(f"- 個別文件: {output_dir}/[combat_dir]/[segment]_{style_name}_commentary.txt")
    
    def save_enhanced_comparison_results(self, all_results, output_dir):
        """
        保存增強版比較結果（支持多目錄結構）
        
        Args:
            all_results (dict): 所有風格的結果
            output_dir (str): 輸出目錄
        """
        comparison_path = os.path.join(output_dir, "enhanced_style_comparison.txt")
        
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write("FightingICE Commentary Style Comparison (Multi-Directory)\n")
            f.write("=" * 60 + "\n\n")
            
            # 收集所有unique的segment組合
            all_segments = set()
            for style_name, results in all_results.items():
                for segment_key in results.keys():
                    all_segments.add(segment_key)
            
            # 按combat目錄和segment組織
            combat_segments = {}
            for segment_key in all_segments:
                parts = segment_key.split('_', 1)
                if len(parts) >= 2:
                    combat_dir = parts[0]
                    segment_name = parts[1]
                    
                    if combat_dir not in combat_segments:
                        combat_segments[combat_dir] = []
                    combat_segments[combat_dir].append((segment_key, segment_name))
            
            # 輸出比較結果
            for combat_dir in sorted(combat_segments.keys()):
                f.write(f"【COMBAT DIRECTORY: {combat_dir.upper()}】\n")
                f.write("=" * 50 + "\n\n")
                
                segments = sorted(combat_segments[combat_dir], key=lambda x: x[1])
                
                for segment_key, segment_name in segments:
                    f.write(f"◆ {segment_name.upper()}\n")
                    
                    # 先顯示frame data（從任一風格中取得）
                    frame_data = None
                    for style_name, results in all_results.items():
                        if segment_key in results:
                            frame_data = results[segment_key]['frame_data']
                            break
                    
                    if frame_data:
                        f.write(f"Frame Data: {frame_data}\n\n")
                        
                        # 顯示各風格的評論
                        for style_name, results in all_results.items():
                            if segment_key in results:
                                f.write(f"{style_name.upper()}: {results[segment_key]['commentary']}\n")
                        
                        f.write("-" * 40 + "\n\n")
                
                f.write("\n")
        
        print(f"\n增強版風格比較結果已保存到: {comparison_path}")

    def display_style_info(self):
        """顯示所有風格的配置信息"""
        print("\n=== 風格配置信息 ===")
        print("=" * 50)
        
        for style in CommentaryStyle:
            config = self.style_configs[style]
            print(f"\n【{style.value.upper()} 風格】")
            print("-" * 30)
            print("系統提示詞:")
            print(config.system_prompt[:200] + "..." if len(config.system_prompt) > 200 else config.system_prompt)
            print()

def main():
    """
    主函數 - 使用範例
    """
    generator = MultiStyleFightingCommentaryGenerator()
    
    print("增強版多風格格鬥遊戲解說生成器")
    print("=" * 50)
    print("1. 掃描並顯示所有可用的combat目錄")
    print("2. 生成單一風格解說（所有目錄）")
    print("3. 生成所有風格解說（所有目錄）")
    print("4. 生成指定目錄的解說")
    print("5. 顯示風格配置信息")
    
    choice = input("\n請選擇功能 (1-5): ").strip()
    
    if choice == "1":
        # 掃描目錄
        base_dir = input("請輸入基礎目錄路徑 (預設 'combat_analysis'): ").strip()
        if not base_dir:
            base_dir = "combat_analysis"
        
        print(f"\n掃描目錄: {base_dir}")
        all_segments = generator.scan_all_segments(base_dir)
        
        if all_segments:
            print(f"\n發現的combat目錄詳情:")
            for combat_dir, segments in all_segments.items():
                print(f"\n【{combat_dir}】")
                for filename, filepath in segments:
                    print(f"  - {filename}")
        
    elif choice == "2":
        # 單一風格生成（所有目錄）
        print("\n可用風格:")
        for i, style in enumerate(CommentaryStyle, 1):
            print(f"{i}. {style.value}")
        
        style_choice = input("\n請選擇風格 (1-4): ").strip()
        base_dir = input("請輸入基礎目錄路徑 (預設 'combat_analysis'): ").strip()
        if not base_dir:
            base_dir = "combat_analysis"
        
        try:
            style_index = int(style_choice) - 1
            selected_style = list(CommentaryStyle)[style_index]
            
            print(f"\n開始生成 {selected_style.value} 風格解說...")
            results = generator.process_all_segments_single_style(selected_style, base_dir)
            print(f"\n完成！共處理了 {len(results)} 個segments")
            
        except (ValueError, IndexError):
            print("無效的選擇！")
            
    elif choice == "3":
        # 所有風格生成（所有目錄）
        base_dir = input("請輸入基礎目錄路徑 (預設 'combat_analysis'): ").strip()
        if not base_dir:
            base_dir = "combat_analysis"
        
        print(f"\n開始生成所有風格解說...")
        all_results = generator.process_all_segments_all_styles(base_dir)
        
        total_processed = sum(len(results) for results in all_results.values())
        print(f"\n所有風格處理完成！總共處理了 {total_processed} 個segments")
        
    elif choice == "4":
        # 指定目錄生成
        base_dir = input("請輸入基礎目錄路徑 (預設 'combat_analysis'): ").strip()
        if not base_dir:
            base_dir = "combat_analysis"
        
        # 先掃描可用目錄
        all_segments = generator.scan_all_segments(base_dir)
        if not all_segments:
            print("未發現任何combat目錄")
            return
        
        print("\n可用的combat目錄:")
        available_dirs = list(all_segments.keys())
        for i, dir_name in enumerate(available_dirs, 1):
            print(f"{i}. {dir_name}")
        
        dir_choices = input("\n請輸入要處理的目錄編號（用逗號分隔，如 1,3,5）: ").strip()
        
        try:
            selected_indices = [int(x.strip()) - 1 for x in dir_choices.split(',')]
            selected_dirs = [available_dirs[i] for i in selected_indices if 0 <= i < len(available_dirs)]
            
            if not selected_dirs:
                print("無效的選擇！")
                return
            
            # 選擇風格
            print("\n可用風格:")
            for i, style in enumerate(CommentaryStyle, 1):
                print(f"{i}. {style.value}")
            
            style_choice = input("\n請選擇風格 (1-4): ").strip()
            
            style_index = int(style_choice) - 1
            selected_style = list(CommentaryStyle)[style_index]
            
            print(f"\n開始處理指定目錄 {selected_dirs}，風格: {selected_style.value}")
            results = generator.process_specific_combat_dirs(selected_dirs, selected_style, base_dir)
            print(f"\n完成！共處理了 {len(results)} 個segments")
            
        except (ValueError, IndexError):
            print("無效的選擇！")
    
    elif choice == "5":
        # 顯示風格配置信息
        generator.display_style_info()
    
    else:
        print("無效的選擇！")

def demo_usage():
    """示範使用方法"""
    print("=== 使用示範 ===")
    
    # 創建生成器
    generator = MultiStyleFightingCommentaryGenerator()
    
    # 示範1: 掃描目錄
    print("\n1. 掃描combat目錄:")
    all_segments = generator.scan_all_segments()
    
    # 示範2: 生成單一風格解說
    print("\n2. 生成激進風格解說:")
    aggressive_results = generator.process_all_segments_single_style(
        CommentaryStyle.AGGRESSIVE
    )
    
    # 示範3: 處理指定目錄
    print("\n3. 處理指定目錄:")
    if all_segments:
        first_combat_dir = list(all_segments.keys())[0]
        selective_results = generator.process_specific_combat_dirs(
            [first_combat_dir], 
            CommentaryStyle.TECHNICAL
        )
    
    # 示範4: 顯示風格信息
    print("\n4. 顯示風格配置:")
    generator.display_style_info()

def batch_process_example():
    """批量處理示例"""
    generator = MultiStyleFightingCommentaryGenerator()
    
    # 自定義配置
    custom_base_dir = "my_combat_data"
    custom_output_dir = "my_commentaries"
    
    # 處理所有風格
    print("開始批量處理所有風格...")
    all_results = generator.process_all_segments_all_styles(
        base_dir=custom_base_dir,
        output_dir=custom_output_dir
    )
    
    # 統計結果
    for style_name, results in all_results.items():
        print(f"{style_name}: {len(results)} segments processed")

if __name__ == "__main__":
    # 執行主程序
    main()
    
    # 可選：執行示範
    # demo_usage()
    
    # 可選：批量處理示例
    # batch_process_example()