import json
from typing import List, Tuple
import os
from datetime import datetime
import glob
import sys

# 修正Windows編碼問題
if sys.platform.startswith('win'):
    import locale
    # 設置UTF-8編碼
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except:
            pass

class FastFrameAnalyzer:
    __slots__ = ()  # 減少記憶體開銷
    
    @staticmethod
    def parse_and_detect_events(file_path: str) -> List[Tuple[int, str]]:
        """一次性解析檔案並偵測事件，返回 (frame, description) tuples"""
        events = []
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
                
                if prev_frame is not None:
                    frame_num = frame['frame num']
                    
                    # HP changes
                    self_hp_diff = frame['self']['hp'] - prev_frame['self']['hp']
                    opp_hp_diff = frame['opponent']['hp'] - prev_frame['opponent']['hp']
                    
                    if self_hp_diff < 0:
                        events.append((frame_num, f"Self took {-self_hp_diff} damage"))
                    if opp_hp_diff < 0:
                        events.append((frame_num, f"Opponent took {-opp_hp_diff} damage"))
                    
                    # Energy gain (>5)
                    energy_diff = frame['self']['energy'] - prev_frame['self']['energy']
                    if energy_diff > 5:
                        events.append((frame_num, f"Self gained {energy_diff} energy"))
                    
                    # State changes
                    if frame['self']['state'] != prev_frame['self']['state']:
                        events.append((frame_num, f"Self changed from {prev_frame['self']['state']} to {frame['self']['state']}"))
                
                prev_frame = frame
        
        return events
    
    @staticmethod
    def events_to_flow(events: List[Tuple[int, str]]) -> str:
        """將事件列表轉換為combat flow字串"""
        if not events:
            return "No significant events"
        return " → ".join(f"Frame {frame}: {desc}" for frame, desc in events)
    
    @staticmethod
    def analyze_file(file_path: str) -> str:
        """分析整個檔案並返回combat flow"""
        events = FastFrameAnalyzer.parse_and_detect_events(file_path)
        return FastFrameAnalyzer.events_to_flow(events)
    
    @staticmethod
    def analyze_segments(file_path: str, segment_size: int = 180) -> List[str]:
        """分段分析並返回combat flow列表"""
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
                
                # 檢測事件 (和parse_and_detect_events相同邏輯)
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
                
                # 檢查是否完成一個segment
                if current_frame_count >= segment_size:
                    flows.append(FastFrameAnalyzer.events_to_flow(current_segment_events))
                    current_segment_events = []
                    current_frame_count = 0
                
                prev_frame = frame
        
        # 處理最後一個不完整的segment
        if current_segment_events:
            flows.append(FastFrameAnalyzer.events_to_flow(current_segment_events))
        
        return flows
    
    @staticmethod
    def save_flows(file_path: str, output_path: str, segment_size: int = 180):
        """直接分析並儲存combat flows到檔案"""
        flows = FastFrameAnalyzer.analyze_segments(file_path, segment_size)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, flow in enumerate(flows, 1):
                f.write(f"Segment {i}:\n{flow}\n\n")

# 簡化的使用函數
def get_combat_flow(file_path: str) -> str:
    """最快速獲取完整combat flow"""
    return FastFrameAnalyzer.analyze_file(file_path)

def get_segment_flows(file_path: str, segment_size: int = 180) -> List[str]:
    """最快速獲取分段combat flows"""
    return FastFrameAnalyzer.analyze_segments(file_path, segment_size)

def save_combat_flows(input_file: str, output_file: str, segment_size: int = 180):
    """最快速分析並儲存"""
    FastFrameAnalyzer.save_flows(input_file, output_file, segment_size)

def safe_print(text: str):
    """安全打印，避免編碼錯誤"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 如果遇到編碼錯誤，使用ASCII編碼並忽略錯誤字符
        print(text.encode('ascii', 'ignore').decode('ascii'))

def process_all_txt_files(input_folder: str = "./blackmambarecord", segment_size: int = 180):
    """處理指定資料夾中的所有 .txt 檔案"""
    
    # 檢查輸入資料夾是否存在
    if not os.path.exists(input_folder):
        safe_print(f"Error: Folder {input_folder} does not exist")
        return
    
    # 獲取所有 .txt 檔案
    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
    
    if not txt_files:
        safe_print(f"No .txt files found in {input_folder}")
        return
    
    safe_print(f"Found {len(txt_files)} .txt files")
    
    # 創建輸出資料夾 (使用時間戳記)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = f"combat_analysis"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        safe_print(f"Created output folder: {output_folder}")
    
    # 處理每個檔案
    for file_index, txt_file in enumerate(txt_files, 1):
        try:
            safe_print(f"Processing: {os.path.basename(txt_file)}")
            
            # 獲取檔案名稱 (不含副檔名)
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            
            # 為每個檔案創建子資料夾 (使用編號)
            folder_name = f"{file_index:02d}"  # 01, 02, 03...
            file_output_folder = os.path.join(output_folder, folder_name)
            segments_folder = os.path.join(file_output_folder, "segments")
            
            os.makedirs(file_output_folder, exist_ok=True)
            os.makedirs(segments_folder, exist_ok=True)
            
            # 1. 獲取完整的 combat flow
            full_flow = get_combat_flow(txt_file)
            
            # 2. 獲取分段的 combat flows
            segment_flows = get_segment_flows(txt_file, segment_size)
            
            # 保存完整流程
            full_flow_path = os.path.join(file_output_folder, "full_combat_flow.txt")
            with open(full_flow_path, "w", encoding="utf-8") as f:
                f.write(full_flow)
            
            # 保存每個segment為獨立文件
            for i, flow in enumerate(segment_flows, 1):
                filename = f"segment_{i:02d}.txt"
                filepath = os.path.join(segments_folder, filename)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(flow)
            
            safe_print(f"✓ {folder_name} ({base_name}) processed successfully ({len(segment_flows)} segments)")
            
        except Exception as e:
            safe_print(f"✗ Error processing {os.path.basename(txt_file)}: {e}")
    
    safe_print(f"\nAll files processed! Results saved in: {output_folder}")
    
    # 創建總結報告
    create_summary_report(output_folder, txt_files)

def create_summary_report(output_folder: str, processed_files: List[str]):
    """創建處理總結報告"""
    summary_path = os.path.join(output_folder, "processing_summary.txt")
    
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Combat Flow Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Files Processed: {len(processed_files)}\n\n")
            
            f.write("File Mapping:\n")
            f.write("-" * 20 + "\n")
            for i, file_path in enumerate(processed_files, 1):
                file_name = os.path.basename(file_path)
                f.write(f"{i:02d}/ -> {file_name}\n")
            
            f.write(f"\nOutput Structure:\n")
            f.write(f"- Each file has its own numbered folder (01/, 02/, 03/...)\n")
            f.write(f"- Each folder contains:\n")
            f.write(f"  - full_combat_flow.txt (complete analysis)\n")
            f.write(f"  - segments/ folder with individual segment files\n")
        
        safe_print(f"Summary report saved: {summary_path}")
    except Exception as e:
        safe_print(f"Error creating summary report: {e}")

# 主執行部分
if __name__ == "__main__":
    # 設置環境變量來處理編碼問題
    if sys.platform.startswith('win'):
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 處理 blackmambarecord 資料夾中的所有 .txt 檔案
    process_all_txt_files("./blackmambarecord", 180)
    
    # 如果你想要處理不同的資料夾或調整 segment 大小，可以這樣做：
    # process_all_txt_files("./your_folder", 120)  # 處理其他資料夾，segment大小為120