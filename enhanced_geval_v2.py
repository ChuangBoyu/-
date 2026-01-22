"""
Enhanced G-Eval Commentary Evaluation System v2
增強版 G-Eval 評論評估系統 v2

改進重點：
1. 從文件名解析 method（因為 JSON 內容可能不一致）
2. 完整的交叉統計（Style × Method）
3. 論文所需的統計量（Mean, SD, SE, 95% CI, n）
4. 更嚴謹的資料驗證
"""

import os
import re
import json
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import scipy.stats as stats

# OpenAI API
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Install with: pip install openai")


# ============================================================
# Enums & Data Classes
# ============================================================

class CommentaryStyle(Enum):
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    TECHNICAL = "technical"
    ENTERTAINMENT = "entertainment"


class EvaluationMethod(Enum):
    LLM = "LLM"      # 大型語言模型 (GPT-4o)
    OURS = "OURS"    # 你的方法 (知識蒸餾)
    SLM = "SLM"      # 小型語言模型 baseline


@dataclass
class ParrotingResult:
    """機械複述檢測結果"""
    is_parroting: bool
    similarity_percentage: float
    has_frame_format: bool
    is_simple_rewrite: bool
    evidence: str
    decision: str


@dataclass
class QualityResult:
    """文本品質評估結果"""
    score: float
    raw_score: int
    reasoning: str
    confidence: float
    cot_steps: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StyleResult:
    """風格一致性評估結果"""
    score: float
    raw_score: int
    reasoning: str
    confidence: float
    target_style: CommentaryStyle = None
    cot_steps: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FullEvaluationResult:
    """完整評估結果"""
    commentary: str
    frame_data: str
    target_style: CommentaryStyle
    method: EvaluationMethod
    
    # 三個評估維度
    parroting_result: ParrotingResult
    quality_result: Optional[QualityResult]
    style_result: Optional[StyleResult]
    
    # 元數據
    timestamp: str = ""
    segment_id: str = ""
    source_file: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "commentary": self.commentary,
            "frame_data": self.frame_data,
            "target_style": self.target_style.value,
            "method": self.method.value,
            "segment_id": self.segment_id,
            "source_file": self.source_file,
            "timestamp": self.timestamp,
            "parroting_check": {
                "is_parroting": self.parroting_result.is_parroting,
                "similarity_percentage": self.parroting_result.similarity_percentage,
                "has_frame_format": self.parroting_result.has_frame_format,
                "is_simple_rewrite": self.parroting_result.is_simple_rewrite,
                "evidence": self.parroting_result.evidence,
                "decision": self.parroting_result.decision
            },
            "quality_evaluation": {
                "score": self.quality_result.score if self.quality_result else None,
                "raw_score": self.quality_result.raw_score if self.quality_result else None,
                "reasoning": self.quality_result.reasoning if self.quality_result else "Skipped due to parroting",
                "confidence": self.quality_result.confidence if self.quality_result else 0.0,
                "cot_steps": self.quality_result.cot_steps if self.quality_result else {}
            },
            "style_evaluation": {
                "score": self.style_result.score if self.style_result else None,
                "raw_score": self.style_result.raw_score if self.style_result else None,
                "reasoning": self.style_result.reasoning if self.style_result else "Skipped due to parroting",
                "confidence": self.style_result.confidence if self.style_result else 0.0,
                "cot_steps": self.style_result.cot_steps if self.style_result else {}
            }
        }


# ============================================================
# 文件名解析器（關鍵改進）
# ============================================================

class FilenameParser:
    """
    從文件名解析 style 和 method
    格式: eval_{segment_id}_{style}_{method}.json
    例如: eval_01_aggressive_llm.json
    """
    
    STYLE_KEYWORDS = {
        "aggressive": CommentaryStyle.AGGRESSIVE,
        "defensive": CommentaryStyle.DEFENSIVE,
        "technical": CommentaryStyle.TECHNICAL,
        "entertainment": CommentaryStyle.ENTERTAINMENT,
    }
    
    METHOD_KEYWORDS = {
        "llm": EvaluationMethod.LLM,
        "ours": EvaluationMethod.OURS,
        "slm": EvaluationMethod.SLM,
    }
    
    @classmethod
    def parse(cls, filename: str) -> Tuple[str, Optional[CommentaryStyle], Optional[EvaluationMethod]]:
        """
        解析文件名
        
        Args:
            filename: 文件名（如 eval_01_aggressive_llm.json）
            
        Returns:
            (segment_id, style, method)
        """
        # 移除副檔名
        name = filename.replace(".json", "").lower()
        parts = name.split("_")
        
        segment_id = None
        style = None
        method = None
        
        # 解析各部分
        for i, part in enumerate(parts):
            # 跳過 "eval" 前綴
            if part == "eval":
                continue
            
            # 檢查是否為 segment_id（數字）
            if part.isdigit():
                segment_id = f"SEGMENT_{part.zfill(2)}"
                continue
            
            # 檢查 style
            if part in cls.STYLE_KEYWORDS:
                style = cls.STYLE_KEYWORDS[part]
                continue
            
            # 檢查 method
            if part in cls.METHOD_KEYWORDS:
                method = cls.METHOD_KEYWORDS[part]
                continue
        
        return segment_id, style, method
    
    @classmethod
    def validate_consistency(cls, filename: str, json_data: Dict) -> Dict[str, Any]:
        """
        驗證文件名與 JSON 內容的一致性
        
        Returns:
            {
                "is_consistent": bool,
                "filename_style": str,
                "json_style": str,
                "filename_method": str,
                "json_method": str,
                "warnings": List[str]
            }
        """
        segment_id, file_style, file_method = cls.parse(filename)
        
        json_style = json_data.get("style", "").lower()
        json_method = json_data.get("method", "").lower()
        
        warnings = []
        
        # 檢查 style 一致性
        if file_style and json_style:
            if file_style.value.lower() != json_style:
                warnings.append(
                    f"Style mismatch: filename='{file_style.value}', json='{json_style}'"
                )
        
        # 檢查 method 一致性
        if file_method and json_method:
            if file_method.value.lower() != json_method:
                warnings.append(
                    f"Method mismatch: filename='{file_method.value}', json='{json_method}'"
                )
        
        return {
            "is_consistent": len(warnings) == 0,
            "filename_style": file_style.value if file_style else None,
            "json_style": json_style,
            "filename_method": file_method.value if file_method else None,
            "json_method": json_method,
            "warnings": warnings,
            # 優先使用文件名的解析結果（因為更可靠）
            "resolved_style": file_style,
            "resolved_method": file_method
        }


# ============================================================
# G-Eval Prompts（與原版相同）
# ============================================================

QUALITY_SYSTEM_PROMPT = """You are an expert evaluator for fighting game commentary. Your task is to assess the TEXT QUALITY of AI-generated commentary using Chain-of-Thought reasoning.

You must evaluate commentary that describes fighting game action sequences. Focus on linguistic quality, NOT style or entertainment value.

EVALUATION CRITERIA:
1. Fluency: Grammar, natural flow, readability
2. Coherence: Logical connection to game context
3. Informativeness: Meaningful content about the action
4. Engagement: Interest level as live commentary

SCORING SCALE (1-5):
1 = Poor: Major errors, incoherent, or meaningless
2 = Below Average: Notable issues affecting comprehension
3 = Average: Acceptable but unremarkable
4 = Good: Well-written with only minor issues
5 = Excellent: Fluent, coherent, engaging, professional quality

OUTPUT FORMAT (JSON):
{
    "step_1_fluency": "Assessment of grammar and flow...",
    "step_2_coherence": "Assessment of logical connection...",
    "step_3_informativeness": "Assessment of content value...",
    "step_4_engagement": "Assessment of interest level...",
    "step_5_overall": "Final synthesis...",
    "score": <1-5>,
    "confidence": <0.0-1.0>,
    "reasoning": "Concise summary of evaluation"
}"""

QUALITY_USER_PROMPT = """Evaluate the TEXT QUALITY of this fighting game commentary.

[Battle Log / Frame Data]:
{frame_data}

[Commentary to Evaluate]:
{commentary}

Perform Chain-of-Thought evaluation and output JSON:"""

STYLE_DEFINITIONS = {
    CommentaryStyle.AGGRESSIVE: {
        "persona": "BLITZ HAMMER",
        "description": "Explosive, high-energy commentator who lives for the CLASH",
        "characteristics": [
            "HIGH INTENSITY language with maximum energy",
            "Power words: BLAST, CRUSH, DOMINATE, DESTROY, ANNIHILATE, DEVASTATE",
            "Focus on: Damage dealt, aggressive pressure, offensive dominance",
            "Tone: Every sentence hits like a knockout punch",
            "Energy: Maximum, almost shouting through text"
        ],
        "positive_indicators": ["caps for emphasis", "exclamations", "power verbs", "offensive focus"],
        "negative_indicators": ["calm tone", "analytical language", "defensive focus", "technical jargon"]
    },
    CommentaryStyle.DEFENSIVE: {
        "persona": "Professor Shield",
        "description": "Methodical, calculating analyst like a chess grandmaster",
        "characteristics": [
            "Calm, measured, analytical tone",
            "Tactical vocabulary: calculated, strategic, positioned, anticipated, countered",
            "Focus on: Defensive positioning, spacing, risk management",
            "Tone: Intellectual and precise, like academic analysis",
            "Energy: Low, contemplative, professorial"
        ],
        "positive_indicators": ["tactical terms", "calm analysis", "spacing mentions", "risk assessment"],
        "negative_indicators": ["excitement", "power words", "aggressive language", "narrative drama"]
    },
    CommentaryStyle.TECHNICAL: {
        "persona": "Dr. Frame Perfect",
        "description": "Technical authority who speaks like a fighting game engine",
        "characteristics": [
            "Precise, data-driven language",
            "Technical terminology: advantage, disadvantage, optimal, punish, recovery, startup",
            "Focus on: Frame data concepts, mechanical execution, optimal choices",
            "Tone: Clinical and exact, educational",
            "Energy: Neutral, focused on accuracy over entertainment"
        ],
        "positive_indicators": ["frame terminology", "mechanical descriptions", "optimization focus", "technical precision"],
        "negative_indicators": ["emotional language", "narrative elements", "casual expressions", "hype"]
    },
    CommentaryStyle.ENTERTAINMENT: {
        "persona": "Captain Hype Story",
        "description": "Master storyteller who transforms matches into epic tales",
        "characteristics": [
            "Narrative/cinematic language: drama, plot twist, hero, epic, legendary",
            "Storytelling elements: character arcs, comebacks, dramatic moments",
            "Focus on: Entertainment value, humor, crowd-pleasing spectacle",
            "Tone: Movie narrator meets stand-up comedian",
            "Energy: High but warm, inviting rather than aggressive"
        ],
        "positive_indicators": ["story elements", "dramatic language", "character references", "entertainment value"],
        "negative_indicators": ["dry analysis", "technical jargon", "pure aggression", "clinical tone"]
    }
}

FEW_SHOT_EXAMPLES = {
    CommentaryStyle.AGGRESSIVE: [
        {
            "commentary": "BOOM! P1 UNLEASHES A DEVASTATING STRIKE, CRUSHING THROUGH DEFENSES AND OBLITERATING 10 HEALTH LIKE A RAGING STORM!",
            "score": 5,
            "reasoning": "Perfect match. Uses caps, sound effects (BOOM!), high-intensity verbs (obliterating, crushing), and focuses purely on damage and dominance."
        },
        {
            "commentary": "In a frantic duel, the opponent fell like a shadow. The crowd roared, only to be stunned. What an avenger!",
            "score": 3,
            "reasoning": "Partial match. High energy, but the tone is more 'Entertainment/Storytelling' (shadow, avenger) than pure Aggressive. Lacks the raw impact of the persona."
        },
        {
            "commentary": "'Frame 1994: Opponent took 10 damage'",
            "score": 1,
            "reasoning": "No match. This is raw log data. Completely lacks emotion, intensity, or aggressive vocabulary."
        }
    ],
    CommentaryStyle.DEFENSIVE: [
        {
            "commentary": "P1 demonstrated calculated spacing, strategically positioned to mitigate risk, effectively anticipating the opponent's advance and creating a counter-opportunity for damage.",
            "score": 5,
            "reasoning": "Perfect match. Uses key tactical vocabulary (calculated, spacing, mitigate risk, counter-opportunity) and maintains a calm, analytical tone."
        },
        {
            "commentary": "Opponent's defensive integrity faltered; poor spacing and timing led to repeated damage.",
            "score": 3,
            "reasoning": "Good match conceptually, but slightly brief. It captures the analytical tone but lacks the depth found in higher-scoring examples."
        },
        {
            "commentary": "Frame 2179: Self changed from DOWN to STAND. Self changed from DOWN to STAND.",
            "score": 1,
            "reasoning": "No match. Repetitive raw data without any analysis or insight into defensive strategy."
        }
    ],
    CommentaryStyle.TECHNICAL: [
        {
            "commentary": "P1 demonstrates rapid state transitions with precise crouch cancels, optimizing meter gain and damage output. Air-state usage appears fluid, but stance shifts risk punish due to recovery vulnerability.",
            "score": 5,
            "reasoning": "Perfect match. Discusses specific mechanics (crouch cancels, meter gain, recovery vulnerability) with clinical precision."
        },
        {
            "commentary": "At frame 1994, P1 feigned weakness, luring the opponent into striking—only to counter with a legendary move, delivering a shocking ten-damage twist!",
            "score": 3,
            "reasoning": "Partial match. Describes a technical event (a counter), but the language is too narrative ('legendary move', 'twist') and lacks technical precision."
        },
        {
            "commentary": "Frame 2718: Self took 5 damage",
            "score": 1,
            "reasoning": "No match. Just a raw data log. No explanation of frame data, advantage, or optimization."
        }
    ],
    CommentaryStyle.ENTERTAINMENT: [
        {
            "commentary": "From the ashes of DOWN, P1 rose to STAND—like a phoenix defying fate, shouting, The fight isn't over yet!",
            "score": 5,
            "reasoning": "Perfect match. Uses strong narrative metaphors ('phoenix defying fate', 'ashes') and focuses on the dramatic arc of the character."
        },
        {
            "commentary": "Amid chaotic crouches and stands, our hero baited defiance, soared to strike, but fate's symphony of twists ducked him down... only to strike... the crowd roared: 'Legend!'",
            "score": 3,
            "reasoning": "Partial match. Attempts storytelling ('hero', 'fate's symphony'), but the flow is disjointed and slightly confusing, reducing the entertainment value."
        },
        {
            "commentary": "Frame 2179: Self changed from DOWN to STAND. Frame 2179: Self changed from DOWN to STAND.",
            "score": 1,
            "reasoning": "No match. Robotic repetition of logs. Zero entertainment value or narrative."
        }
    ]
}

def get_style_system_prompt(style: CommentaryStyle) -> str:
    """生成包含 Few-Shot 的 System Prompt"""
    style_info = STYLE_DEFINITIONS[style]
    examples = FEW_SHOT_EXAMPLES[style]
    
    characteristics = "\n".join(f"- {c}" for c in style_info["characteristics"])
    positive = ", ".join(style_info["positive_indicators"])
    negative = ", ".join(style_info["negative_indicators"])
    
    few_shot_str = "\n".join([
        f"Input: \"{ex['commentary']}\"\nOutput Score: {ex['score']}\nReasoning: {ex['reasoning']}\n"
        for ex in examples
    ])
    
    return f"""You are an expert evaluator for fighting game commentary style consistency using Chain-of-Thought reasoning.

You must provide a stylistic evaluation of the commentary describing action sequences in fighting games. The focus is on stylistic expression, not the quality of the language.

TARGET STYLE: "{style_info['persona']}" ({style.value.upper()})
Description: {style_info['description']}

STYLE CHARACTERISTICS:
{characteristics}

POSITIVE INDICATORS (should be present): {positive}
NEGATIVE INDICATORS (should be absent): {negative}

SCORING SCALE (1-5):
1 = No match: Completely different or opposite style (e.g., Raw Data Logs)
2 = Weak match: Some elements but lacks core characteristics
3 = Partial match: Moderate alignment but inconsistent tone or mixed styles
4 = Good match: Clearly matches with minor deviations
5 = Perfect match: Unmistakably this style's representative

FEW-SHOT EXAMPLES (Use these as calibration):
{few_shot_str}

OUTPUT FORMAT (JSON):
{{
    "step_1_positive_check": "Which positive indicators are present...",
    "step_2_negative_check": "Which negative indicators appear...",
    "step_3_tone_assessment": "Overall tone alignment compared to examples...",
    "step_4_energy_match": "Energy level match...",
    "step_5_overall": "Final style consistency judgment...",
    "score": <1-5>,
    "confidence": <0.0-1.0>,
    "reasoning": "Concise summary of style evaluation"
}}"""

STYLE_USER_PROMPT = """Evaluate the STYLE CONSISTENCY of the following commentary against the target style, strictly referring to the FEW-SHOT EXAMPLES provided in the system instruction for calibration.

[Commentary to Evaluate]:
"{commentary}"

[Target Style]: {style_name} ({style_persona})

Instruction:
1. Analyze the commentary based on the style definition.
2. Compare it with the provided Few-Shot Examples (High/Mid/Low scores) to determine where it fits on the scale.
3. Perform the step-by-step reasoning (Positive/Negative checks, Tone, Energy) as defined in the output format.
4. Output the final JSON object.
"""


# ============================================================
# Main Evaluator Class
# ============================================================

class EnhancedGEvalEvaluator:
    """增強版 G-Eval 評估器"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5.2-2025-12-11"):
        if not OPENAI_AVAILABLE:
            raise ImportError("Please install openai: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        self.hyperparams = {
            "temperature": 0.1,
            "max_completion_tokens": 1000,
            "top_p": 0.9
        }
        
        self.parroting_threshold = 70
        self.parroting_frame_threshold = 50
        self.parroting_rewrite_threshold = 40
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """計算兩個文本的 Jaccard 相似度"""
        def normalize_text(text: str) -> str:
            text = text.lower()
            text = re.sub(r'\b(the|a|an)\b', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = ' '.join(text.split())
            return text
        
        norm_text1 = normalize_text(text1)
        norm_text2 = normalize_text(text2)
        
        words1 = set(norm_text1.split())
        words2 = set(norm_text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return (len(intersection) / len(union)) * 100
    
    def detect_parroting(self, frame_data: str, commentary: str) -> ParrotingResult:
        """檢測機械複述"""
        similarity = self.calculate_similarity(frame_data, commentary)
        
        frame_pattern = r'frame\s+\d+\s*:'
        has_frame_format = bool(re.search(frame_pattern, commentary.lower()))
        
        simple_rewrites = [
            r'^(the\s+)?frame\s+\d+',
            r'frame\s+\d+.*took\s+\d+\s+damage',
            r'at\s+frame\s+\d+',
            r'during\s+frame\s+\d+',
        ]
        is_simple_rewrite = any(re.search(pattern, commentary.lower()) for pattern in simple_rewrites)
        
        is_parroting = (
            similarity > self.parroting_threshold or
            (similarity > self.parroting_frame_threshold and has_frame_format) or
            (similarity > self.parroting_rewrite_threshold and is_simple_rewrite)
        )
        
        evidence = []
        if similarity > self.parroting_threshold:
            evidence.append(f"High text similarity: {similarity:.1f}%")
        if has_frame_format:
            evidence.append("Retained 'Frame X:' format from log")
        if is_simple_rewrite:
            evidence.append("Simple rewrite without natural language transformation")
        
        return ParrotingResult(
            is_parroting=is_parroting,
            similarity_percentage=round(similarity, 1),
            has_frame_format=has_frame_format,
            is_simple_rewrite=is_simple_rewrite,
            evidence=" | ".join(evidence) if evidence else "No parroting detected",
            decision="FAIL - Parroting" if is_parroting else "PASS"
        )
    
    def _call_gpt(self, system_prompt: str, user_prompt: str, 
                  use_logprobs: bool = False) -> Tuple[str, Optional[Dict]]:
        """調用 GPT API"""
        try:
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **self.hyperparams
            }
            
            if use_logprobs:
                params["logprobs"] = True
                params["top_logprobs"] = 5
            
            response = self.client.chat.completions.create(**params)
            
            text = response.choices[0].message.content
            logprobs_data = response.choices[0].logprobs if use_logprobs else None
            
            return text, logprobs_data
            
        except Exception as e:
            print(f"    ✗ API Error: {e}")
            time.sleep(20)
            return self._call_gpt(system_prompt, user_prompt, use_logprobs)
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """解析 JSON 回應"""
        text = text.strip()
        
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        score_match = re.search(r'"?score"?\s*:\s*(\d+)', text, re.IGNORECASE)
        confidence_match = re.search(r'"?confidence"?\s*:\s*(0?\.\d+|1\.0)', text, re.IGNORECASE)
        
        return {
            "score": int(score_match.group(1)) if score_match else 3,
            "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
            "reasoning": text,
            "parse_failed": True
        }
    
    def _calculate_weighted_score(self, logprobs_data) -> Optional[float]:
        """從 logprobs 計算加權分數"""
        if not logprobs_data or not logprobs_data.content:
            return None
        
        score_tokens = ["1", "2", "3", "4", "5"]
        
        for token_info in reversed(logprobs_data.content):
            if token_info.token.strip() in score_tokens:
                if token_info.top_logprobs:
                    probs = {}
                    for top_lp in token_info.top_logprobs:
                        if top_lp.token.strip() in score_tokens:
                            probs[int(top_lp.token.strip())] = np.exp(top_lp.logprob)
                    
                    if probs:
                        total_prob = sum(probs.values())
                        weighted_sum = sum(score * prob for score, prob in probs.items())
                        return weighted_sum / total_prob if total_prob > 0 else None
                break
        
        return None
    
    def evaluate_quality(self, frame_data: str, commentary: str) -> QualityResult:
        """評估文本品質"""
        user_prompt = QUALITY_USER_PROMPT.format(
            frame_data=frame_data,
            commentary=commentary
        )
        
        text, logprobs = self._call_gpt(QUALITY_SYSTEM_PROMPT, user_prompt, use_logprobs=True)
        result = self._parse_json_response(text)
        
        weighted_score = self._calculate_weighted_score(logprobs)
        raw_score = result.get("score", 3)
        
        return QualityResult(
            score=weighted_score if weighted_score else float(raw_score),
            raw_score=raw_score,
            reasoning=result.get("reasoning", ""),
            confidence=result.get("confidence", 0.5),
            cot_steps={
                "step_1_fluency": result.get("step_1_fluency", ""),
                "step_2_coherence": result.get("step_2_coherence", ""),
                "step_3_informativeness": result.get("step_3_informativeness", ""),
                "step_4_engagement": result.get("step_4_engagement", ""),
                "step_5_overall": result.get("step_5_overall", "")
            }
        )
    
    def evaluate_style(self, commentary: str, target_style: CommentaryStyle) -> StyleResult:
        """評估風格一致性"""
        style_info = STYLE_DEFINITIONS[target_style]
        system_prompt = get_style_system_prompt(target_style)
        
        user_prompt = STYLE_USER_PROMPT.format(
            commentary=commentary,
            style_name=target_style.value.upper(),
            style_persona=style_info["persona"]
        )
        
        text, logprobs = self._call_gpt(system_prompt, user_prompt, use_logprobs=True)
        result = self._parse_json_response(text)
        
        weighted_score = self._calculate_weighted_score(logprobs)
        raw_score = result.get("score", 3)
        
        return StyleResult(
            score=weighted_score if weighted_score else float(raw_score),
            raw_score=raw_score,
            reasoning=result.get("reasoning", ""),
            confidence=result.get("confidence", 0.5),
            target_style=target_style,
            cot_steps={
                "step_1_positive_check": result.get("step_1_positive_check", ""),
                "step_2_negative_check": result.get("step_2_negative_check", ""),
                "step_3_tone_assessment": result.get("step_3_tone_assessment", ""),
                "step_4_energy_match": result.get("step_4_energy_match", ""),
                "step_5_overall": result.get("step_5_overall", "")
            }
        )
    
    def evaluate_full(self, 
                      frame_data: str, 
                      commentary: str, 
                      target_style: CommentaryStyle,
                      method: EvaluationMethod = EvaluationMethod.OURS,
                      segment_id: str = "",
                      source_file: str = "") -> FullEvaluationResult:
        """完整評估"""
        timestamp = datetime.now().isoformat()
        
        parroting_result = self.detect_parroting(frame_data, commentary)
        
        if parroting_result.is_parroting:
            print(f"    ⚠️  Parroting detected (similarity: {parroting_result.similarity_percentage}%) - Score: 1")
            
            quality_result = QualityResult(
                score=1.0,
                raw_score=1,
                reasoning=f"Parroting detected: {parroting_result.evidence}",
                confidence=0.95,
                cot_steps={"parroting_override": True}
            )
            
            style_result = StyleResult(
                score=1.0,
                raw_score=1,
                reasoning=f"Parroting detected: {parroting_result.evidence}",
                confidence=0.95,
                target_style=target_style,
                cot_steps={"parroting_override": True}
            )
        else:
            quality_result = self.evaluate_quality(frame_data, commentary)
            style_result = self.evaluate_style(commentary, target_style)
        
        return FullEvaluationResult(
            commentary=commentary,
            frame_data=frame_data,
            target_style=target_style,
            method=method,
            parroting_result=parroting_result,
            quality_result=quality_result,
            style_result=style_result,
            timestamp=timestamp,
            segment_id=segment_id,
            source_file=source_file
        )


# ============================================================
# 改進版批次評估器
# ============================================================

class BatchEvaluatorV2:
    """
    改進版批次評估處理器
    
    關鍵改進：
    1. 從文件名解析 style 和 method（比 JSON 內容更可靠）
    2. 驗證文件名與 JSON 內容的一致性
    3. 記錄所有不一致的情況
    """
    
    def __init__(self, evaluator: EnhancedGEvalEvaluator):
        self.evaluator = evaluator
        self.results = []
        self.consistency_warnings = []
    
    def load_and_validate_files(self, input_dir: str) -> List[Dict[str, Any]]:
        """
        載入並驗證所有評估文件
        
        Returns:
            驗證後的文件列表
        """
        files_data = []
        
        # 只處理 eval_*.json 文件
        json_files = sorted([
            f for f in os.listdir(input_dir) 
            if f.endswith('.json') and f.startswith('eval_')
        ])
        
        print(f"\n{'='*60}")
        print(f"Loading and validating {len(json_files)} files...")
        print(f"{'='*60}")
        
        for filename in json_files:
            file_path = os.path.join(input_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 驗證一致性
                validation = FilenameParser.validate_consistency(filename, data)
                
                if not validation["is_consistent"]:
                    for warning in validation["warnings"]:
                        self.consistency_warnings.append({
                            "file": filename,
                            "warning": warning
                        })
                        print(f"  ⚠️  {filename}: {warning}")
                
                # 使用文件名解析的結果（更可靠）
                files_data.append({
                    "filename": filename,
                    "filepath": file_path,
                    "data": data,
                    "resolved_style": validation["resolved_style"],
                    "resolved_method": validation["resolved_method"],
                    "validation": validation
                })
                
                print(f"  ✓ {filename} -> Style: {validation['resolved_style'].value}, Method: {validation['resolved_method'].value}")
                
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
        
        if self.consistency_warnings:
            print(f"\n⚠️  Found {len(self.consistency_warnings)} consistency warnings!")
            print("   Note: Using FILENAME parsing results (more reliable than JSON content)")
        
        return files_data
    
    def evaluate_from_files(self, input_dir: str, 
                           output_file: str = "evaluation_results_v2.json") -> Dict[str, Any]:
        """從文件批次評估"""
        files_data = self.load_and_validate_files(input_dir)
        
        results = {
            "metadata": {
                "evaluation_method": "G-Eval Enhanced v2 (Quality + Style + Parroting)",
                "model": self.evaluator.model,
                "parroting_threshold": self.evaluator.parroting_threshold,
                "timestamp": datetime.now().isoformat(),
                "total_files": len(files_data),
                "consistency_warnings": self.consistency_warnings
            },
            "evaluations": []
        }
        
        print(f"\n{'='*60}")
        print("Starting G-Eval Evaluation...")
        print(f"{'='*60}")
        
        for i, file_info in enumerate(files_data):
            filename = file_info["filename"]
            data = file_info["data"]
            
            # 使用解析後的 style 和 method
            target_style = file_info["resolved_style"]
            method = file_info["resolved_method"]
            
            print(f"\n[{i+1}/{len(files_data)}] {filename}")
            print(f"  Style: {target_style.value} | Method: {method.value}")
            
            frame_data = data.get("frame_data", "")
            commentary = data.get("commentary", "")
            segment_id = data.get("segment_id", "")
            
            # 執行評估
            eval_result = self.evaluator.evaluate_full(
                frame_data=frame_data,
                commentary=commentary,
                target_style=target_style,
                method=method,
                segment_id=segment_id,
                source_file=filename
            )
            
            result_dict = eval_result.to_dict()
            result_dict["source_file"] = filename
            result_dict["complexity"] = data.get("complexity", {})
            results["evaluations"].append(result_dict)
            
            print(f"  ✓ Quality: {eval_result.quality_result.score:.2f} | Style: {eval_result.style_result.score:.2f}")
            
            time.sleep(1)
        
        # 保存結果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        return results


# ============================================================
# 論文級統計分析
# ============================================================

class PaperStatisticsGenerator:
    """
    論文級統計報告生成器
    
    生成內容：
    1. 整體統計（Overall）
    2. 按方法分組（By Method: LLM, OURS, SLM）
    3. 按風格分組（By Style: Aggressive, Defensive, Technical, Entertainment）
    4. 交叉統計（Style × Method）
    5. 完整統計量（Mean, SD, SE, 95% CI, n, Min, Max）
    """
    
    @staticmethod
    def calculate_statistics(scores: List[float]) -> Dict[str, Any]:
        """
        計算論文所需的完整統計量
        
        Returns:
            {
                "n": 樣本數,
                "mean": 平均值,
                "std": 標準差,
                "se": 標準誤差,
                "ci_95_lower": 95% CI 下界,
                "ci_95_upper": 95% CI 上界,
                "min": 最小值,
                "max": 最大值,
                "median": 中位數
            }
        """
        if not scores or len(scores) == 0:
            return {
                "n": 0,
                "mean": None,
                "std": None,
                "se": None,
                "ci_95_lower": None,
                "ci_95_upper": None,
                "min": None,
                "max": None,
                "median": None
            }
        
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1) if n > 1 else 0  # 樣本標準差
        se = std / np.sqrt(n) if n > 0 else 0  # 標準誤差
        
        # 95% 信賴區間
        if n > 1:
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=se)
            ci_lower, ci_upper = ci
        else:
            ci_lower, ci_upper = mean, mean
        
        return {
            "n": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "se": round(se, 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "median": round(np.median(scores), 4)
        }
    
    @classmethod
    def generate_full_statistics(cls, results: Dict[str, Any], 
                                output_file: str = "paper_statistics.json") -> Dict[str, Any]:
        """
        生成完整的論文級統計報告
        """
        evaluations = results.get("evaluations", [])
        
        # 初始化數據收集器
        data = {
            "overall": {
                "quality_scores": [],
                "style_scores": [],
                "parroting_count": 0,
                "total": 0
            },
            "by_method": defaultdict(lambda: {
                "quality_scores": [],
                "style_scores": [],
                "parroting_count": 0
            }),
            "by_style": defaultdict(lambda: {
                "quality_scores": [],
                "style_scores": [],
                "parroting_count": 0
            }),
            "by_style_method": defaultdict(lambda: defaultdict(lambda: {
                "quality_scores": [],
                "style_scores": [],
                "parroting_count": 0
            }))
        }
        
        # 收集數據
        for eval_item in evaluations:
            method = eval_item.get("method", "UNKNOWN")
            style = eval_item.get("target_style", "UNKNOWN")
            
            is_parroting = eval_item.get("parroting_check", {}).get("is_parroting", False)
            quality_score = eval_item.get("quality_evaluation", {}).get("score")
            style_score = eval_item.get("style_evaluation", {}).get("score")
            
            # Overall
            data["overall"]["total"] += 1
            if is_parroting:
                data["overall"]["parroting_count"] += 1
            if quality_score is not None:
                data["overall"]["quality_scores"].append(quality_score)
            if style_score is not None:
                data["overall"]["style_scores"].append(style_score)
            
            # By Method
            if is_parroting:
                data["by_method"][method]["parroting_count"] += 1
            if quality_score is not None:
                data["by_method"][method]["quality_scores"].append(quality_score)
            if style_score is not None:
                data["by_method"][method]["style_scores"].append(style_score)
            
            # By Style
            if is_parroting:
                data["by_style"][style]["parroting_count"] += 1
            if quality_score is not None:
                data["by_style"][style]["quality_scores"].append(quality_score)
            if style_score is not None:
                data["by_style"][style]["style_scores"].append(style_score)
            
            # By Style × Method (交叉統計)
            if is_parroting:
                data["by_style_method"][style][method]["parroting_count"] += 1
            if quality_score is not None:
                data["by_style_method"][style][method]["quality_scores"].append(quality_score)
            if style_score is not None:
                data["by_style_method"][style][method]["style_scores"].append(style_score)
        
        # 生成統計報告
        report = {
            "metadata": {
                **results.get("metadata", {}),
                "statistics_generated_at": datetime.now().isoformat()
            },
            "overall": {
                "total_evaluations": data["overall"]["total"],
                "parroting_count": data["overall"]["parroting_count"],
                "parroting_rate": round(
                    data["overall"]["parroting_count"] / max(data["overall"]["total"], 1) * 100, 2
                ),
                "quality": cls.calculate_statistics(data["overall"]["quality_scores"]),
                "style": cls.calculate_statistics(data["overall"]["style_scores"])
            },
            "by_method": {},
            "by_style": {},
            "by_style_method": {}
        }
        
        # By Method 統計
        for method in ["LLM", "OURS", "SLM"]:
            method_data = data["by_method"].get(method, {
                "quality_scores": [],
                "style_scores": [],
                "parroting_count": 0
            })
            n = len(method_data["quality_scores"])
            report["by_method"][method] = {
                "n": n,
                "parroting_count": method_data["parroting_count"],
                "parroting_rate": round(
                    method_data["parroting_count"] / max(n, 1) * 100, 2
                ) if n > 0 else 0,
                "quality": cls.calculate_statistics(method_data["quality_scores"]),
                "style": cls.calculate_statistics(method_data["style_scores"])
            }
        
        # By Style 統計
        for style in ["aggressive", "defensive", "technical", "entertainment"]:
            style_data = data["by_style"].get(style, {
                "quality_scores": [],
                "style_scores": [],
                "parroting_count": 0
            })
            n = len(style_data["quality_scores"])
            report["by_style"][style] = {
                "n": n,
                "parroting_count": style_data["parroting_count"],
                "parroting_rate": round(
                    style_data["parroting_count"] / max(n, 1) * 100, 2
                ) if n > 0 else 0,
                "quality": cls.calculate_statistics(style_data["quality_scores"]),
                "style": cls.calculate_statistics(style_data["style_scores"])
            }
        
        # By Style × Method 交叉統計
        for style in ["aggressive", "defensive", "technical", "entertainment"]:
            report["by_style_method"][style] = {}
            for method in ["LLM", "OURS", "SLM"]:
                sm_data = data["by_style_method"].get(style, {}).get(method, {
                    "quality_scores": [],
                    "style_scores": [],
                    "parroting_count": 0
                })
                n = len(sm_data["quality_scores"])
                report["by_style_method"][style][method] = {
                    "n": n,
                    "parroting_count": sm_data["parroting_count"],
                    "quality": cls.calculate_statistics(sm_data["quality_scores"]),
                    "style": cls.calculate_statistics(sm_data["style_scores"])
                }
        
        # 保存報告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印摘要
        cls._print_summary(report)
        
        return report
    
    @classmethod
    def _print_summary(cls, report: Dict[str, Any]):
        """打印統計摘要"""
        print("\n" + "=" * 80)
        print("PAPER-READY STATISTICS SUMMARY")
        print("=" * 80)
        
        # Overall
        overall = report["overall"]
        print(f"\n📊 OVERALL (n={overall['total_evaluations']})")
        print(f"   Parroting Rate: {overall['parroting_rate']}%")
        print(f"   Quality: {overall['quality']['mean']:.3f} ± {overall['quality']['std']:.3f} "
              f"(95% CI: [{overall['quality']['ci_95_lower']:.3f}, {overall['quality']['ci_95_upper']:.3f}])")
        print(f"   Style:   {overall['style']['mean']:.3f} ± {overall['style']['std']:.3f} "
              f"(95% CI: [{overall['style']['ci_95_lower']:.3f}, {overall['style']['ci_95_upper']:.3f}])")
        
        # By Method
        print(f"\n📈 BY METHOD")
        print("-" * 80)
        print(f"{'Method':<10} {'n':>5} {'Quality Mean':>14} {'Quality SD':>12} {'Style Mean':>12} {'Style SD':>10}")
        print("-" * 80)
        for method in ["LLM", "OURS", "SLM"]:
            m = report["by_method"][method]
            q = m["quality"]
            s = m["style"]
            if q["n"] > 0:
                print(f"{method:<10} {q['n']:>5} {q['mean']:>14.3f} {q['std']:>12.3f} {s['mean']:>12.3f} {s['std']:>10.3f}")
        
        # By Style
        print(f"\n🎨 BY STYLE")
        print("-" * 80)
        print(f"{'Style':<15} {'n':>5} {'Quality Mean':>14} {'Quality SD':>12} {'Style Mean':>12} {'Style SD':>10}")
        print("-" * 80)
        for style in ["aggressive", "defensive", "technical", "entertainment"]:
            st = report["by_style"][style]
            q = st["quality"]
            s = st["style"]
            if q["n"] > 0:
                print(f"{style.capitalize():<15} {q['n']:>5} {q['mean']:>14.3f} {q['std']:>12.3f} {s['mean']:>12.3f} {s['std']:>10.3f}")
        
        # Style × Method Cross Table
        print(f"\n📋 STYLE × METHOD CROSS TABLE (Quality Scores)")
        print("-" * 60)
        print(f"{'Style':<15} {'LLM':>12} {'OURS':>12} {'SLM':>12}")
        print("-" * 60)
        for style in ["aggressive", "defensive", "technical", "entertainment"]:
            row = f"{style.capitalize():<15}"
            for method in ["LLM", "OURS", "SLM"]:
                sm = report["by_style_method"][style][method]
                if sm["quality"]["n"] > 0:
                    row += f" {sm['quality']['mean']:>11.3f}"
                else:
                    row += f" {'N/A':>11}"
            print(row)
        
        print(f"\n📋 STYLE × METHOD CROSS TABLE (Style Scores)")
        print("-" * 60)
        print(f"{'Style':<15} {'LLM':>12} {'OURS':>12} {'SLM':>12}")
        print("-" * 60)
        for style in ["aggressive", "defensive", "technical", "entertainment"]:
            row = f"{style.capitalize():<15}"
            for method in ["LLM", "OURS", "SLM"]:
                sm = report["by_style_method"][style][method]
                if sm["style"]["n"] > 0:
                    row += f" {sm['style']['mean']:>11.3f}"
                else:
                    row += f" {'N/A':>11}"
            print(row)
        
        print("\n" + "=" * 80)
    
    @classmethod
    def export_for_latex(cls, report: Dict[str, Any], 
                        output_file: str = "latex_tables.txt") -> str:
        """
        導出 LaTeX 格式的表格
        """
        latex = []
        
        # Table 1: Overall Results by Method
        latex.append("% Table 1: Overall Results by Method")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Evaluation Results by Generation Method}")
        latex.append("\\label{tab:results_by_method}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append("Method & n & Quality (M ± SD) & Style (M ± SD) & Parroting \\\\")
        latex.append("\\midrule")
        
        for method in ["LLM", "OURS", "SLM"]:
            m = report["by_method"][method]
            q = m["quality"]
            s = m["style"]
            if q["n"] > 0:
                latex.append(
                    f"{method} & {q['n']} & "
                    f"${q['mean']:.2f} \\pm {q['std']:.2f}$ & "
                    f"${s['mean']:.2f} \\pm {s['std']:.2f}$ & "
                    f"{m['parroting_rate']:.1f}\\% \\\\"
                )
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex.append("")
        
        # Table 2: Style × Method Cross Table (Quality)
        latex.append("% Table 2: Quality Scores by Style and Method")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Text Quality Scores by Style and Method}")
        latex.append("\\label{tab:quality_cross}")
        latex.append("\\begin{tabular}{lccc}")
        latex.append("\\toprule")
        latex.append("Style & LLM & OURS & SLM \\\\")
        latex.append("\\midrule")
        
        for style in ["aggressive", "defensive", "technical", "entertainment"]:
            row = f"{style.capitalize()}"
            for method in ["LLM", "OURS", "SLM"]:
                sm = report["by_style_method"][style][method]
                if sm["quality"]["n"] > 0:
                    row += f" & ${sm['quality']['mean']:.2f}$"
                else:
                    row += " & --"
            row += " \\\\"
            latex.append(row)
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex.append("")
        
        # Table 3: Style × Method Cross Table (Style Consistency)
        latex.append("% Table 3: Style Consistency Scores by Style and Method")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Style Consistency Scores by Style and Method}")
        latex.append("\\label{tab:style_cross}")
        latex.append("\\begin{tabular}{lccc}")
        latex.append("\\toprule")
        latex.append("Style & LLM & OURS & SLM \\\\")
        latex.append("\\midrule")
        
        for style in ["aggressive", "defensive", "technical", "entertainment"]:
            row = f"{style.capitalize()}"
            for method in ["LLM", "OURS", "SLM"]:
                sm = report["by_style_method"][style][method]
                if sm["style"]["n"] > 0:
                    row += f" & ${sm['style']['mean']:.2f}$"
                else:
                    row += " & --"
            row += " \\\\"
            latex.append(row)
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        latex_str = "\n".join(latex)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        
        print(f"\n✓ LaTeX tables saved to: {output_file}")
        
        return latex_str
    
    @classmethod
    def export_for_csv(cls, report: Dict[str, Any], 
                      output_prefix: str = "statistics") -> None:
        """
        導出 CSV 格式（方便用 Excel 或其他工具分析）
        """
        import csv
        
        # CSV 1: By Method
        with open(f"{output_prefix}_by_method.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Method", "n", 
                "Quality_Mean", "Quality_SD", "Quality_SE", "Quality_CI_Lower", "Quality_CI_Upper",
                "Style_Mean", "Style_SD", "Style_SE", "Style_CI_Lower", "Style_CI_Upper",
                "Parroting_Count", "Parroting_Rate"
            ])
            
            for method in ["LLM", "OURS", "SLM"]:
                m = report["by_method"][method]
                q = m["quality"]
                s = m["style"]
                writer.writerow([
                    method, q["n"],
                    q["mean"], q["std"], q["se"], q["ci_95_lower"], q["ci_95_upper"],
                    s["mean"], s["std"], s["se"], s["ci_95_lower"], s["ci_95_upper"],
                    m["parroting_count"], m["parroting_rate"]
                ])
        
        # CSV 2: By Style
        with open(f"{output_prefix}_by_style.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Style", "n", 
                "Quality_Mean", "Quality_SD", "Quality_SE", "Quality_CI_Lower", "Quality_CI_Upper",
                "Style_Mean", "Style_SD", "Style_SE", "Style_CI_Lower", "Style_CI_Upper",
                "Parroting_Count", "Parroting_Rate"
            ])
            
            for style in ["aggressive", "defensive", "technical", "entertainment"]:
                st = report["by_style"][style]
                q = st["quality"]
                s = st["style"]
                writer.writerow([
                    style, q["n"],
                    q["mean"], q["std"], q["se"], q["ci_95_lower"], q["ci_95_upper"],
                    s["mean"], s["std"], s["se"], s["ci_95_lower"], s["ci_95_upper"],
                    st["parroting_count"], st["parroting_rate"]
                ])
        
        # CSV 3: Style × Method Cross (Quality)
        with open(f"{output_prefix}_cross_quality.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Style", "LLM", "OURS", "SLM"])
            
            for style in ["aggressive", "defensive", "technical", "entertainment"]:
                row = [style]
                for method in ["LLM", "OURS", "SLM"]:
                    sm = report["by_style_method"][style][method]
                    row.append(sm["quality"]["mean"] if sm["quality"]["n"] > 0 else "")
                writer.writerow(row)
        
        # CSV 4: Style × Method Cross (Style Consistency)
        with open(f"{output_prefix}_cross_style.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Style", "LLM", "OURS", "SLM"])
            
            for style in ["aggressive", "defensive", "technical", "entertainment"]:
                row = [style]
                for method in ["LLM", "OURS", "SLM"]:
                    sm = report["by_style_method"][style][method]
                    row.append(sm["style"]["mean"] if sm["style"]["n"] > 0 else "")
                writer.writerow(row)
        
        print(f"\n✓ CSV files saved with prefix: {output_prefix}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced G-Eval Commentary Evaluator v2")
    parser.add_argument("--input", "-i", type=str, help="Input directory with eval_*.json files")
    parser.add_argument("--output", "-o", type=str, default="evaluation_results_v2.json", help="Output file")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-5.2-2025-12-11", help="Model to use")
    parser.add_argument("--stats-only", type=str, help="Only generate statistics from existing results file")
    parser.add_argument("--export-latex", action="store_true", help="Export LaTeX tables")
    parser.add_argument("--export-csv", action="store_true", help="Export CSV files")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample data")
    
    args = parser.parse_args()
    
    if args.demo:
        print("=" * 60)
        print("ENHANCED G-EVAL v2 DEMO")
        print("=" * 60)
        
        # 演示文件名解析
        test_filenames = [
            "eval_01_aggressive_llm.json",
            "eval_02_defensive_ours.json",
            "eval_03_technical_slm.json",
            "eval_04_entertainment_llm.json"
        ]
        
        print("\nFilename Parsing Demo:")
        for fn in test_filenames:
            seg_id, style, method = FilenameParser.parse(fn)
            print(f"  {fn}")
            print(f"    -> Segment: {seg_id}, Style: {style.value if style else 'None'}, Method: {method.value if method else 'None'}")
        
        print("\nTo run actual evaluation, provide --api-key or set OPENAI_API_KEY")
        return
    
    # 只生成統計
    if args.stats_only:
        if not os.path.exists(args.stats_only):
            print(f"Error: Results file '{args.stats_only}' not found")
            return
        
        with open(args.stats_only, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        stats_file = args.stats_only.replace(".json", "_statistics.json")
        report = PaperStatisticsGenerator.generate_full_statistics(results, stats_file)
        
        if args.export_latex:
            PaperStatisticsGenerator.export_for_latex(
                report, 
                args.stats_only.replace(".json", "_latex.txt")
            )
        
        if args.export_csv:
            PaperStatisticsGenerator.export_for_csv(
                report,
                args.stats_only.replace(".json", "")
            )
        
        return
    
    # 實際評估
    api_key = args.api_key or os.getenv("")
    if not api_key:
        print("Error: API key required. Use --api-key or set OPENAI_API_KEY")
        return
    
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input directory '{args.input}' not found")
            return
        
        print("=" * 60)
        print("STARTING ENHANCED G-EVAL v2 EVALUATION")
        print("=" * 60)
        
        evaluator = EnhancedGEvalEvaluator(api_key=api_key, model=args.model)
        batch_evaluator = BatchEvaluatorV2(evaluator)
        
        results = batch_evaluator.evaluate_from_files(args.input, args.output)
        
        # 生成統計
        stats_file = args.output.replace(".json", "_statistics.json")
        report = PaperStatisticsGenerator.generate_full_statistics(results, stats_file)
        
        if args.export_latex:
            PaperStatisticsGenerator.export_for_latex(
                report, 
                args.output.replace(".json", "_latex.txt")
            )
        
        if args.export_csv:
            PaperStatisticsGenerator.export_for_csv(
                report,
                args.output.replace(".json", "")
            )
    else:
        print("No input specified. Use --input <directory>, --stats-only <file>, or --demo")


if __name__ == "__main__":
    main()