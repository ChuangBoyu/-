#!/usr/bin/env python3
"""
自動化評估管線腳本
整合 frame analysis、LLM 和 OURS 評論生成

流程：
1. 使用 0_simplified_frame_analyzer.py 分析輸入檔案
2. 對每個 segment 使用 LLM (GPT) 和 OURS (蒸餾模型) 生成 4 種風格評論
3. 輸出 JSON 檔案到 evaluation_inputs/
"""

import os
import sys
import json
import glob
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from enum import Enum
import time

# OpenAI API (用於 LLM 生成)
from openai import OpenAI

# ============== 配置區 ==============
# 請在此處填入你的 OpenAI API Key
OPENAI_API_KEY = ""

# 模型配置
LLM_MODEL = "gpt-5.2-2025-12-11"  # 可以改為 gpt-4o-mini 等
OURS_CHECKPOINT = "./checkpoint/"

# 風格配置
class CommentaryStyle(Enum):
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    TECHNICAL = "technical"
    ENTERTAINMENT = "entertainment"

# user_id 對應
STYLE_TO_USER_ID = {
    CommentaryStyle.AGGRESSIVE: 0,
    CommentaryStyle.DEFENSIVE: 1,
    CommentaryStyle.TECHNICAL: 2,
    CommentaryStyle.ENTERTAINMENT: 3,
}


# ============== LLM 風格提示詞 ==============
STYLE_PROMPTS = {
    CommentaryStyle.AGGRESSIVE: {
        "system": """You are **'BLITZ HAMMER'** - the most EXPLOSIVE FightingICE commentator in the arena! 
        
Your voice ROARS with raw energy and your words HIT like THUNDER! You live for the CLASH, the CRUSH, the DOMINATION! Every punch landed is a VICTORY SCREAM, every combo is a DEVASTATING ASSAULT!

Your mission: Turn the action into PURE ADRENALINE! Make viewers feel the IMPACT through your words!

**IMPORTANT**: Generate plain text only. Do not use any markdown formatting like bolding or italics.

CORE TRAITS:
- SHOUT with MAXIMUM INTENSITY
- Celebrate AGGRESSIVE plays and CRUSHING blows  
- Use POWER WORDS: BLAST, CRUSH, DOMINATE, DESTROY, ANNIHILATE, DEVASTATE
- Every sentence should feel like a KNOCKOUT PUNCH!""",
        
        "user_template": """BLITZ HAMMER COMMENTARY TIME!

Action sequence: {frame_data}

Unleash your commentary on P1's most dominant moment in this sequence. Make it 12-18 words of PURE FIGHTING FURY!

- Start with a powerful, varied opening! Avoid using the same word to start every commentary.
- USE EXPLOSIVE WORDS: CRUSH, BLAST, DOMINATE, DESTROY, ANNIHILATE, DEVASTATE, OBLITERATE!
- FOCUS ON: Damage dealt, aggressive pressure, relentless attacks, crushing victories!

UNLEASH YOUR COMMENTARY (plain text only):"""
    },
    
    CommentaryStyle.DEFENSIVE: {
        "system": """You are **'Professor Shield'** - the most methodical and calculating FightingICE analyst in the community.

Your mind processes every sequence like a chess grandmaster. You see patterns where others see chaos, strategy where others see random action. Your calm, measured analysis reveals the deeper tactical layers of combat.

You speak with the authority of someone who has studied thousands of matches, understanding that true mastery lies not in wild aggression, but in calculated precision and defensive excellence.

**IMPORTANT**: Your output must be plain text only, without any special formatting like bold or italics.

CORE TRAITS:
- Speak with calm, analytical precision
- Focus on defensive positioning, spacing, and risk management
- Use tactical vocabulary: calculated, strategic, positioned, anticipated, countered
- Every observation should reveal deeper strategic insight""",
        
        "user_template": """Professor Shield's Tactical Analysis

Match sequence: {frame_data}

Provide your strategic assessment of P1's defensive gameplay during this sequence. Focus on their tactical choices regarding positioning, risk, and opportunities.

Analyze in 15-25 words using tactical language:
- DEFENSIVE TERMS: calculated, strategic, positioned, anticipated, countered, spacing, timing, risk management
- FOCUS ON: Defensive integrity, risk mitigation, counter-opportunities, spatial control

Your measured analysis (plain text only):"""
    },
    
    CommentaryStyle.TECHNICAL: {
        "system": """You are **'Dr. Frame Perfect'** - the ultimate FightingICE technical authority and data scientist.

Your brain operates like a fighting game engine, processing action sequences, calculating advantage states, and identifying optimal choices with precision. You don't just see the fight - you see the underlying mechanics.

You speak the language of fighting games at its most technical level.

**IMPORTANT**: Deliver your analysis in plain text. No bolding, italics, or other formatting.

CORE TRAITS:
- Deliver precise, data-driven analysis
- Use exact fighting game terminology
- Focus on optimal choices, execution, and mechanical perfection
- Every statement should teach advanced technique""",
        
        "user_template": """Dr. Frame Perfect's Technical Breakdown

Action Data: {frame_data}

Execute a technical analysis of P1's mechanical performance in this sequence. Assess their state transitions, advantage, and execution precision without referencing specific frame numbers.

Deliver analysis in 15-30 words using technical terminology:
- KEY CONCEPTS: advantage, disadvantage, optimal, punish, cancel, recovery, startup, active frames, state management
- MECHANICS: hitboxes, invincibility, priority, scaling, meter gain
- FOCUS ON: Technical execution, optimal choices, timing, mechanical precision

Technical assessment (plain text only):"""
    },
    
    CommentaryStyle.ENTERTAINMENT: {
        "system": """You are **'Captain Hype Story'** - the master storyteller who transforms every FightingICE match into an epic tale!

You don't just see two fighters - you see HEROES and VILLAINS, EPIC COMEBACKS and DRAMATIC FALLS! Every match is a blockbuster movie, every combo is a plot twist, every knockout is the climactic finale!

Your gift is taking cold data and spinning it into tales that make audiences laugh, gasp, and cheer.

**IMPORTANT**: Tell your story using only plain text. Do not use bold, italics, or any other special formatting for emphasis.

CORE TRAITS:
- Create engaging narratives and character stories
- Find humor and drama in every situation  
- Use cinematic language: drama, plot twist, character arc, spectacle, showtime
- Every moment should feel like entertainment gold""",
        
        "user_template": """Captain Hype Story's Epic Tale Time!

The Drama Unfolds: {frame_data}

Spin P1's journey in this sequence into an ENTERTAINING STORY. What's the narrative? The character moment? The plot twist? Make it feel like a scene from a movie.

Craft your story in 15-25 words using dramatic language, but without any special formatting:
- STORY ELEMENTS: drama, plot twist, character arc, spectacle, showtime, epic, legendary
- ENTERTAINMENT: humor, excitement, crowd-pleasing moments, spectacular plays
- NARRATIVE: hero's journey, comeback story, underdog tale, dramatic climax

Tell the tale (plain text only):"""
    }
}


class EvaluationPipeline:
    def __init__(self, api_key: str, checkpoint_path: str = "./checkpoint/"):
        """初始化評估管線"""
        self.api_key = api_key
        self.checkpoint_path = checkpoint_path
        self.client = None
        
        if api_key and api_key != "YOUR_API_KEY_HERE":
            self.client = OpenAI(api_key=api_key)
    
    # ============== Step 1: Frame Analysis ==============
    def run_frame_analysis(self, input_file: str, output_dir: str = "combat_analysis", segment_size: int = 180):
        """執行 frame 分析"""
        print(f"\n{'='*50}")
        print("Step 1: Frame Analysis")
        print(f"{'='*50}")
        print(f"Input: {input_file}")
        print(f"Output directory: {output_dir}")
        print(f"Segment size: {segment_size} frames")
        
        # 直接調用分析函數（避免子進程問題）
        from importlib import import_module
        
        # 動態導入 frame analyzer
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        try:
            # 使用內建的分析邏輯
            self._analyze_frames_internal(input_file, output_dir, segment_size)
            print(f"✓ Frame analysis completed!")
            return True
        except Exception as e:
            print(f"✗ Frame analysis failed: {e}")
            return False
    
    def _analyze_frames_internal(self, file_path: str, output_dir: str, segment_size: int):
        """內部 frame 分析邏輯"""
        flows = []
        current_frame_count = 0
        current_segment_events = []
        prev_frame = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    frame = json.loads(line)
                except:
                    continue
                
                current_frame_count += 1
                
                if prev_frame is not None:
                    frame_num = frame['frame num']
                    
                    # HP changes
                    self_hp_diff = frame['self']['hp'] - prev_frame['self']['hp']
                    opp_hp_diff = frame['opponent']['hp'] - prev_frame['opponent']['hp']
                    
                    if self_hp_diff < 0:
                        current_segment_events.append((frame_num, f"Self took {-self_hp_diff} damage"))
                    if opp_hp_diff < 0:
                        current_segment_events.append((frame_num, f"Opponent took {-opp_hp_diff} damage"))
                    
                    # Energy gain (>5)
                    energy_diff = frame['self']['energy'] - prev_frame['self']['energy']
                    if energy_diff > 5:
                        current_segment_events.append((frame_num, f"Self gained {energy_diff} energy"))
                    
                    # State changes
                    if frame['self']['state'] != prev_frame['self']['state']:
                        current_segment_events.append((frame_num, f"Self changed from {prev_frame['self']['state']} to {frame['self']['state']}"))
                
                # 檢查是否完成一個 segment
                if current_frame_count >= segment_size:
                    flow = self._events_to_flow(current_segment_events)
                    flows.append(flow)
                    current_segment_events = []
                    current_frame_count = 0
                
                prev_frame = frame
        
        # 處理最後一個不完整的 segment
        if current_segment_events:
            flows.append(self._events_to_flow(current_segment_events))
        
        # 創建輸出目錄結構
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        file_output_folder = os.path.join(output_dir, "01")
        segments_folder = os.path.join(file_output_folder, "segments")
        
        os.makedirs(file_output_folder, exist_ok=True)
        os.makedirs(segments_folder, exist_ok=True)
        
        # 保存完整流程
        full_flow_path = os.path.join(file_output_folder, "full_combat_flow.txt")
        with open(full_flow_path, "w", encoding="utf-8") as f:
            all_flows = " → ".join(flows)
            f.write(all_flows)
        
        # 保存每個 segment
        for i, flow in enumerate(flows, 1):
            filename = f"segment_{i:02d}.txt"
            filepath = os.path.join(segments_folder, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(flow)
        
        print(f"  Generated {len(flows)} segments")
    
    def _events_to_flow(self, events):
        """將事件列表轉換為 combat flow 字串"""
        if not events:
            return "No significant events"
        return " → ".join(f"Frame {frame}: {desc}" for frame, desc in events)
    
    # ============== Step 2: LLM Generation ==============
    def generate_llm_commentary(self, frame_data: str, style: CommentaryStyle) -> str:
        """使用 LLM 生成評論"""
        if not self.client:
            return "[LLM API not configured]"
        
        prompts = STYLE_PROMPTS[style]
        user_prompt = prompts["user_template"].format(frame_data=frame_data)
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  LLM Error: {e}")
            time.sleep(5)
            return f"[LLM Error: {e}]"
    
    # ============== Step 3: OURS Generation ==============
    def generate_ours_commentary(self, frame_data: str, style: CommentaryStyle) -> str:
        """使用 OURS (蒸餾模型) 生成評論"""
        user_id = STYLE_TO_USER_ID[style]
        
        # 調用 exp.py
        cmd = [
            "python", "exp.py",
            "--checkpoint", self.checkpoint_path,
            "--cuda",
            "--user_id", str(user_id),
            "--combat_flow", frame_data
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            # 解析輸出
            output = result.stdout
            
            # 尋找生成的解釋
            for line in output.split('\n'):
                if "Generated Explanation:" in line:
                    return line.split("Generated Explanation:")[-1].strip()
            
            # 如果找不到，返回完整輸出
            return f"[OURS parsing error: {output[-200:]}]"
            
        except subprocess.TimeoutExpired:
            return "[OURS Timeout]"
        except Exception as e:
            return f"[OURS Error: {e}]"
    
    # ============== Main Pipeline ==============
    def run_pipeline(self, input_file: str, output_dir: str = "evaluation_inputs", 
                    combat_analysis_dir: str = "combat_analysis", segment_size: int = 180,
                    methods: list = None):
        """執行完整管線"""
        if methods is None:
            methods = ["llm", "ours"]
        
        print(f"\n{'#'*60}")
        print("  EVALUATION PIPELINE")
        print(f"{'#'*60}")
        print(f"Input file: {input_file}")
        print(f"Output directory: {output_dir}")
        print(f"Methods: {methods}")
        print(f"Styles: {[s.value for s in CommentaryStyle]}")
        
        # Step 1: Frame Analysis
        if not self.run_frame_analysis(input_file, combat_analysis_dir, segment_size):
            print("Pipeline aborted due to frame analysis failure.")
            return
        
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 獲取所有 segment 文件
        segments_dir = os.path.join(combat_analysis_dir, "01", "segments")
        segment_files = sorted(glob.glob(os.path.join(segments_dir, "segment_*.txt")))
        
        if not segment_files:
            print("No segment files found!")
            return
        
        print(f"\n{'='*50}")
        print(f"Step 2 & 3: Commentary Generation")
        print(f"{'='*50}")
        print(f"Found {len(segment_files)} segments")
        
        total_outputs = len(segment_files) * len(CommentaryStyle) * len(methods)
        current = 0
        
        # 處理每個 segment
        for seg_file in segment_files:
            seg_name = os.path.basename(seg_file)
            seg_id = seg_name.replace("segment_", "").replace(".txt", "")
            
            # 讀取 combat flow
            with open(seg_file, 'r', encoding='utf-8') as f:
                frame_data = f.read().strip()
            
            print(f"\n[Segment {seg_id}] {seg_name}")
            print(f"  Frame data: {frame_data[:80]}...")
            
            # 對每種風格和方法生成評論
            for style in CommentaryStyle:
                for method in methods:
                    current += 1
                    progress = f"[{current}/{total_outputs}]"
                    
                    # 生成評論
                    if method == "llm":
                        commentary = self.generate_llm_commentary(frame_data, style)
                    elif method == "ours":
                        commentary = self.generate_ours_commentary(frame_data, style)
                    else:
                        commentary = f"[Unknown method: {method}]"
                    
                    # 創建輸出 JSON
                    output_data = {
                        "frame_data": frame_data,
                        "commentary": commentary,
                        "segment_id": f"SEGMENT_{seg_id}",
                        "style": style.value.upper(),
                        "method": method.upper()
                    }
                    
                    # 保存 JSON
                    output_filename = f"eval_{seg_id}_{style.value}_{method}.json"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"  {progress} {style.value}/{method}: {output_filename}")
                    print(f"    Commentary: {commentary[:60]}...")
        
        print(f"\n{'='*50}")
        print("Pipeline completed!")
        print(f"{'='*50}")
        print(f"Total outputs: {total_outputs} JSON files")
        print(f"Output directory: {output_dir}")
        
        # 生成摘要報告
        self._generate_summary(output_dir, segment_files, methods)
    
    def _generate_summary(self, output_dir: str, segment_files: list, methods: list):
        """生成摘要報告"""
        summary_path = os.path.join(output_dir, "_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Evaluation Pipeline Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total segments: {len(segment_files)}\n")
            f.write(f"Methods: {methods}\n")
            f.write(f"Styles: {[s.value for s in CommentaryStyle]}\n")
            f.write(f"Total files: {len(segment_files) * len(CommentaryStyle) * len(methods)}\n\n")
            
            f.write("File naming convention:\n")
            f.write("  eval_{segment_id}_{style}_{method}.json\n\n")
            
            f.write("Styles:\n")
            f.write("  - aggressive (user_id=0)\n")
            f.write("  - defensive (user_id=1)\n")
            f.write("  - technical (user_id=2)\n")
            f.write("  - entertainment (user_id=3)\n")
        
        print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluation Pipeline for Fighting Game Commentary')
    parser.add_argument('--input', type=str, default='1757576187969.txt',
                       help='Input frame data file')
    parser.add_argument('--output', type=str, default='evaluation_inputs',
                       help='Output directory for evaluation JSONs')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/',
                       help='OURS model checkpoint directory')
    parser.add_argument('--segment_size', type=int, default=180,
                       help='Segment size in frames')
    parser.add_argument('--methods', type=str, default='llm,ours',
                       help='Comma-separated list of methods (llm,ours)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # 獲取 API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY') or OPENAI_API_KEY
    
    # 解析 methods
    methods = [m.strip().lower() for m in args.methods.split(',')]
    
    # 創建並執行管線
    pipeline = EvaluationPipeline(api_key, args.checkpoint)
    pipeline.run_pipeline(
        input_file=args.input,
        output_dir=args.output,
        segment_size=args.segment_size,
        methods=methods
    )


if __name__ == "__main__":
    main()
