import os  
import torch  
import argparse  
from transformers import T5Tokenizer  
from utils.utils import DynamicExpGenerator, now_time  

import warnings  
  
# 抑制所有FutureWarning  
warnings.filterwarnings("ignore", category=FutureWarning)  
  
# 抑制Hugging Face Hub警告  
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  
parser = argparse.ArgumentParser(description='Combat Commentary Generation')  
parser.add_argument('--model_version', type=str, default='google/flan-t5-base',  
                    help='model version')  
parser.add_argument('--checkpoint', type=str, default='./checkpoint/',  
                    help='directory to load the trained model')  
parser.add_argument('--cuda', action='store_true',  
                    help='use CUDA')  
parser.add_argument('--exp_len', type=int, default=100,  
                    help='maximum length of generated explanation')  
parser.add_argument('--user_id', type=int, default=0,  
                    help='user ID (0-3): 0=aggressive, 1=defensive, 2=technical, 3=entertainment')  
parser.add_argument('--item_id', type=int, default=0,  
                    help='item ID (combat sequence)')  
parser.add_argument('--combat_flow', type=str, default=None,  
                    help='actual combat flow text for content-based generation')
parser.add_argument('--debug', action='store_true',  
                    help='enable debug mode to trace input processing')  
args = parser.parse_args()  
  
print('-' * 40 + 'INFERENCE ARGUMENTS' + '-' * 40)  
for arg in vars(args):  
    print('{:40} {}'.format(arg, getattr(args, arg)))  
print('-' * 40 + 'INFERENCE ARGUMENTS' + '-' * 40)  
  
device = torch.device('cuda' if args.cuda else 'cpu')  
  
# 加載訓練好的模型  
model_path = os.path.join(args.checkpoint, 'model.pt')  
if not os.path.exists(model_path):  
    raise FileNotFoundError(f"Model not found at {model_path}")  
  
print(now_time() + f'Loading trained model from {model_path}')  
model = torch.load(model_path, map_location=device, weights_only=False)  
model.eval()  
  
# 加載tokenizer  
print(now_time() + f'Loading tokenizer: {args.model_version}')  
tokenizer = T5Tokenizer.from_pretrained(args.model_version)  
  
# 創建動態解釋生成器  
generator = DynamicExpGenerator(model, tokenizer, device, args.exp_len, debug=args.debug) 
  
# 生成解釋  
if args.combat_flow:  
    # 使用實際戰鬥內容生成  
    print(now_time() + f'Generating explanation with combat flow for user_{args.user_id}')  
    explanation = generator.generate_explanation_with_content(args.user_id, args.combat_flow)  
    print('-' * 40 + 'GENERATED EXPLANATION (CONTENT-BASED)' + '-' * 40)  
    print(f"User Type: {args.user_id}")  
    print(f"Combat Flow: {args.combat_flow[:100]}...")  
    print(f"Generated Explanation: {explanation}")  
else:  
    # 使用ID生成（向後兼容）  
    print(now_time() + f'Generating explanation for user_{args.user_id} item_{args.item_id}')  
    explanation = generator.generate_explanation(args.user_id, args.item_id)  
    print('-' * 40 + 'GENERATED EXPLANATION (ID-BASED)' + '-' * 40)  
    print(f"User Type: {args.user_id}")  
    print(f"Item ID: {args.item_id}")  
    print(f"Generated Explanation: {explanation}")  
  
print('-' * 40 + 'COMPLETED' + '-' * 40)  
  
# 互動模式  
def interactive_mode():  
    print(now_time() + 'Entering interactive mode. Type "quit" to exit.')  
    print("Commands:")  
    print("  id: user_id,item_id (e.g., 0,5)")  
    print("  content: user_id|combat_flow_text")  
    print("  quit: exit")  
      
    while True:  
        try:  
            user_input = input("\nEnter command: ").strip()  
            if user_input.lower() == 'quit':  
                break  
              
            if user_input.startswith('id:'):  
                # ID-based generation  
                _, params = user_input.split(':', 1)  
                user_id, item_id = map(int, params.split(','))  
                if user_id not in [0, 1, 2, 3]:  
                    print("Invalid user_id. Must be 0-3.")  
                    continue  
                explanation = generator.generate_explanation(user_id, item_id)  
                print(f"\nUser Type {user_id} Commentary (ID-based): {explanation}")  
                  
            elif user_input.startswith('content:'):  
                # Content-based generation  
                _, params = user_input.split(':', 1)  
                user_id, combat_flow = params.split('|', 1)  
                user_id = int(user_id)  
                if user_id not in [0, 1, 2, 3]:  
                    print("Invalid user_id. Must be 0-3.")  
                    continue  
                explanation = generator.generate_explanation_with_content(user_id, combat_flow)  
                print(f"\nUser Type {user_id} Commentary (Content-based): {explanation}")  
            else:  
                print("Invalid command format.")  
                  
        except ValueError:  
            print("Invalid input format.")  
        except KeyboardInterrupt:  
            break  
      
    print(now_time() + 'Exiting interactive mode.')  
  
if __name__ == "__main__":  
    # 如果沒有指定特定參數，進入互動模式  
    if not args.combat_flow and args.user_id == 0 and args.item_id == 0:  
        interactive_mode()