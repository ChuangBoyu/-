#!/usr/bin/env python3
"""
SLM Generator - Flan T5-base Zero-shot Commentary Generation

使用原始的 google/flan-t5-base 模型（無微調）進行格鬥遊戲評論生成。
這是作為對照組，與 LLM (GPT) 和 OURS (蒸餾模型) 進行比較。

使用方式:
    # 單次生成
    python 2_slm_generator.py --combat_flow "Frame 47: Opponent took 10 damage" --style aggressive
    
    # 批量處理（整合到管線）
    python 2_slm_generator.py --batch --input_dir combat_analysis/01/segments --output_dir evaluation_inputs
    
    # 互動模式
    python 2_slm_generator.py --interactive
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime
from enum import Enum
import warnings

# 抑制警告
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


# ============== 風格配置 ==============
class CommentaryStyle(Enum):
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    TECHNICAL = "technical"
    ENTERTAINMENT = "entertainment"

# user_id 對應（與 OURS 模型一致）
STYLE_TO_USER_ID = {
    CommentaryStyle.AGGRESSIVE: 0,
    CommentaryStyle.DEFENSIVE: 1,
    CommentaryStyle.TECHNICAL: 2,
    CommentaryStyle.ENTERTAINMENT: 3,
}

# Zero-shot 提示詞模板
STYLE_PROMPTS = {
    CommentaryStyle.AGGRESSIVE: {
        "prefix": "Generate an aggressive and exciting fighting game commentary.",
        "instruction": "Be energetic, use powerful words like CRUSH, BLAST, DOMINATE. Focus on damage and attacks.",
        "example_style": "explosive, intense, high-energy"
    },
    CommentaryStyle.DEFENSIVE: {
        "prefix": "Generate a calm and analytical fighting game commentary.",
        "instruction": "Focus on defensive strategy, positioning, spacing, and tactical decisions.",
        "example_style": "measured, strategic, analytical"
    },
    CommentaryStyle.TECHNICAL: {
        "prefix": "Generate a technical fighting game commentary.",
        "instruction": "Focus on frame data, optimal punishes, state transitions, and mechanical execution.",
        "example_style": "precise, data-driven, technical"
    },
    CommentaryStyle.ENTERTAINMENT: {
        "prefix": "Generate an entertaining and dramatic fighting game commentary.",
        "instruction": "Tell a story, create drama, make it feel like a movie scene.",
        "example_style": "dramatic, narrative, cinematic"
    }
}


class FlanT5SLMGenerator:
    """Flan T5-base SLM 評論生成器"""
    
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = None, max_length: int = 100):
        """
        初始化生成器
        
        Args:
            model_name: Hugging Face 模型名稱
            device: 'cuda' 或 'cpu'，None 則自動檢測
            max_length: 生成的最大長度
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # 自動檢測設備
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        
        # 載入模型和 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def _build_prompt(self, combat_flow: str, style: CommentaryStyle) -> str:
        """
        構建 zero-shot 提示詞
        
        Args:
            combat_flow: 戰鬥流程描述
            style: 評論風格
            
        Returns:
            構建好的提示詞
        """
        style_config = STYLE_PROMPTS[style]
        
        # 構建提示詞（T5 風格）
        prompt = f"""{style_config['prefix']}

Style: {style_config['example_style']}
Instructions: {style_config['instruction']}

Combat action: {combat_flow}

Commentary:"""
        
        return prompt
    
    def generate(self, combat_flow: str, style: CommentaryStyle, 
                 num_beams: int = 4, temperature: float = 0.8,
                 do_sample: bool = True, top_p: float = 0.9) -> str:
        """
        生成評論
        
        Args:
            combat_flow: 戰鬥流程描述
            style: 評論風格
            num_beams: beam search 的 beam 數量
            temperature: 生成溫度
            do_sample: 是否使用採樣
            top_p: nucleus sampling 的 p 值
            
        Returns:
            生成的評論
        """
        prompt = self._build_prompt(combat_flow, style)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            if do_sample:
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
        
        # Decode
        commentary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return commentary.strip()
    
    def generate_all_styles(self, combat_flow: str) -> dict:
        """
        生成所有風格的評論
        
        Args:
            combat_flow: 戰鬥流程描述
            
        Returns:
            字典，包含所有風格的評論
        """
        results = {}
        for style in CommentaryStyle:
            results[style.value] = self.generate(combat_flow, style)
        return results
    
    def batch_process(self, input_dir: str, output_dir: str, 
                      file_pattern: str = "segment_*.txt"):
        """
        批量處理 segment 文件
        
        Args:
            input_dir: 輸入目錄（包含 segment 文件）
            output_dir: 輸出目錄
            file_pattern: 文件匹配模式
        """
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 獲取所有 segment 文件
        segment_files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
        
        if not segment_files:
            print(f"No files found matching {file_pattern} in {input_dir}")
            return
        
        print(f"Found {len(segment_files)} segment files")
        
        total = len(segment_files) * len(CommentaryStyle)
        current = 0
        
        for seg_file in segment_files:
            seg_name = os.path.basename(seg_file)
            seg_id = seg_name.replace("segment_", "").replace(".txt", "")
            
            # 讀取 combat flow
            with open(seg_file, 'r', encoding='utf-8') as f:
                combat_flow = f.read().strip()
            
            print(f"\n[Segment {seg_id}]")
            print(f"  Combat flow: {combat_flow[:60]}...")
            
            # 對每種風格生成評論
            for style in CommentaryStyle:
                current += 1
                
                # 生成評論
                commentary = self.generate(combat_flow, style)
                
                # 創建輸出 JSON
                output_data = {
                    "frame_data": combat_flow,
                    "commentary": commentary,
                    "segment_id": f"SEGMENT_{seg_id}",
                    "style": style.value.upper(),
                    "method": "SLM"
                }
                
                # 保存 JSON
                output_filename = f"eval_{seg_id}_{style.value}_slm.json"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"  [{current}/{total}] {style.value}: {commentary[:50]}...")
        
        print(f"\n{'='*50}")
        print(f"Batch processing completed!")
        print(f"Output directory: {output_dir}")
        print(f"Total files generated: {total}")


def interactive_mode(generator: FlanT5SLMGenerator):
    """互動模式"""
    print("\n" + "="*50)
    print("Flan T5-base SLM Generator - Interactive Mode")
    print("="*50)
    print("\nCommands:")
    print("  <style>|<combat_flow>  - Generate commentary")
    print("  all|<combat_flow>      - Generate all styles")
    print("  styles                 - Show available styles")
    print("  quit                   - Exit")
    print("\nStyles: aggressive, defensive, technical, entertainment")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                break
            
            if user_input.lower() == 'styles':
                for style in CommentaryStyle:
                    config = STYLE_PROMPTS[style]
                    print(f"\n[{style.value.upper()}]")
                    print(f"  {config['prefix']}")
                continue
            
            # 解析輸入
            if '|' not in user_input:
                print("Invalid format. Use: <style>|<combat_flow>")
                continue
            
            style_str, combat_flow = user_input.split('|', 1)
            style_str = style_str.strip().lower()
            combat_flow = combat_flow.strip()
            
            if style_str == 'all':
                # 生成所有風格
                results = generator.generate_all_styles(combat_flow)
                for style_name, commentary in results.items():
                    print(f"\n[{style_name.upper()}]")
                    print(f"  {commentary}")
            else:
                # 生成單一風格
                try:
                    style = CommentaryStyle(style_str)
                    commentary = generator.generate(combat_flow, style)
                    print(f"\n[{style.value.upper()}] {commentary}")
                except ValueError:
                    print(f"Unknown style: {style_str}")
                    print("Available: aggressive, defensive, technical, entertainment")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description='Flan T5-base SLM Commentary Generator')
    
    # 模式選擇
    parser.add_argument('--interactive', action='store_true',
                       help='Enter interactive mode')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process segment files')
    
    # 單次生成參數
    parser.add_argument('--combat_flow', type=str, default=None,
                       help='Combat flow text for single generation')
    parser.add_argument('--style', type=str, default='aggressive',
                       choices=['aggressive', 'defensive', 'technical', 'entertainment'],
                       help='Commentary style')
    
    # 批量處理參數
    parser.add_argument('--input_dir', type=str, default='combat_analysis/01/segments',
                       help='Input directory for batch processing')
    parser.add_argument('--output_dir', type=str, default='evaluation_inputs',
                       help='Output directory for results')
    
    # 模型參數
    parser.add_argument('--model', type=str, default='google/flan-t5-base',
                       help='Model name (default: google/flan-t5-base)')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum generation length')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA if available')
    
    # 生成參數
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Generation temperature')
    parser.add_argument('--num_beams', type=int, default=4,
                       help='Number of beams for beam search')
    parser.add_argument('--no_sample', action='store_true',
                       help='Disable sampling (use greedy/beam search only)')
    
    args = parser.parse_args()
    
    # 設定設備
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    
    # 創建生成器
    generator = FlanT5SLMGenerator(
        model_name=args.model,
        device=device,
        max_length=args.max_length
    )
    
    if args.interactive:
        # 互動模式
        interactive_mode(generator)
    
    elif args.batch:
        # 批量處理模式
        generator.batch_process(args.input_dir, args.output_dir)
    
    elif args.combat_flow:
        # 單次生成模式
        style = CommentaryStyle(args.style)
        commentary = generator.generate(
            args.combat_flow, 
            style,
            num_beams=args.num_beams,
            temperature=args.temperature,
            do_sample=not args.no_sample
        )
        
        print(f"\n{'='*50}")
        print(f"Style: {style.value.upper()}")
        print(f"Combat Flow: {args.combat_flow}")
        print(f"{'='*50}")
        print(f"Commentary: {commentary}")
    
    else:
        # 默認：顯示幫助
        parser.print_help()
        print("\n" + "="*50)
        print("Examples:")
        print("="*50)
        print("\n# Single generation:")
        print('python 2_slm_generator.py --combat_flow "Frame 47: Opponent took 10 damage" --style aggressive')
        print("\n# Batch processing:")
        print("python 2_slm_generator.py --batch --input_dir combat_analysis/01/segments --output_dir evaluation_inputs")
        print("\n# Interactive mode:")
        print("python 2_slm_generator.py --interactive")


if __name__ == "__main__":
    main()
