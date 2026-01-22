import os  
import torch  
import argparse  
import csv
from transformers import T5Tokenizer  
from model.module import Solomon  
from utils.utils import ExpDataLoader, ExpBatchify, now_time  
  
parser = argparse.ArgumentParser(description='RDRec Combat Commentary Training')  
parser.add_argument('--data_dir', type=str, default='./combat_datasets/',  
                    help='directory containing combat_xx folders')  
parser.add_argument('--model_version', type=str, default='google/flan-t5-base',  
                    help='model version: google/flan-t5-small, google/flan-t5-base, etc.')  
parser.add_argument('--task_num', type=int, default=1,  
                    help='task number (1 for explanation only)')  
parser.add_argument('--prompt_num', type=int, default=3,  
                    help='prompts per task')  
parser.add_argument('--lr', type=float, default=0.00005,  
                    help='learning rate')  
parser.add_argument('--epochs', type=int, default=200,  
                    help='upper epoch limit')  
parser.add_argument('--batch_size', type=int, default=16,  
                    help='batch size')  
parser.add_argument('--cuda', action='store_true',  
                    help='use CUDA')  
parser.add_argument('--log_interval', type=int, default=200,  
                    help='report interval')  
parser.add_argument('--checkpoint', type=str, default='./checkpoint/',  
                    help='directory to save the final model')  
parser.add_argument('--endure_times', type=int, default=10,  
                    help='the maximum endure times of loss increasing on validation')  
parser.add_argument('--exp_len', type=int, default=200,  
                    help='the maximum length of an explanation')  
args = parser.parse_args()  
  
print('-' * 40 + 'ARGUMENTS' + '-' * 40)  
for arg in vars(args):  
    print('{:40} {}'.format(arg, getattr(args, arg)))  
print('-' * 40 + 'ARGUMENTS' + '-' * 40)  
  
# CUDA 檢查  
if torch.cuda.is_available():  
    if not args.cuda:  
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')  
device = torch.device('cuda' if args.cuda else 'cpu')  
  
if not os.path.exists(args.checkpoint):  
    os.makedirs(args.checkpoint)  
model_path = os.path.join(args.checkpoint, 'model.pt')

# 建立 loss 紀錄文件
loss_log_path = os.path.join(args.checkpoint, 'training_loss.csv')
with open(loss_log_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'step', 'train_loss', 'val_loss'])
  
###############################################################################  
# Load data  
###############################################################################  
  
print(now_time() + 'Loading data from combat datasets')  
tokenizer = T5Tokenizer.from_pretrained(args.model_version)  
exp_corpus = ExpDataLoader(args.data_dir)  
  
# 創建批處理器，傳入id2item映射以支持實際戰鬥內容  
train_iterator = ExpBatchify(exp_corpus.train, tokenizer, args.exp_len, args.batch_size, exp_corpus.id2item)  
valid_iterator = ExpBatchify(exp_corpus.valid, tokenizer, args.exp_len, args.batch_size, exp_corpus.id2item) if exp_corpus.valid else None  
  
###############################################################################  
# Build the model  
###############################################################################  
  
print(now_time() + f'Loading model: {args.model_version}')  
model = Solomon.from_pretrained(args.model_version)  
model.init_prompt(args.task_num, args.prompt_num, device)  
model.to(device)  
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  
  
###############################################################################  
# Training code  
###############################################################################  

def log_loss(epoch, step, train_loss, val_loss=None):
    """紀錄 loss 到 CSV 文件"""
    with open(loss_log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, step, train_loss, val_loss if val_loss is not None else ''])

def train(epoch):  
    # Turn on training mode which enables dropout.  
    model.train()  
    text_loss = 0.  
    total_sample = 0  
      
    for step in range(train_iterator.total_step):  
        task, source, source_mask, whole_word, target = train_iterator.next_batch()  
        task = task.to(device)  
        source = source.to(device)  
        source_mask = source_mask.to(device)  
        whole_word = whole_word.to(device)  
        target = target.to(device)  
          
        # Starting each batch, we detach the hidden state from how it was previously produced.  
        # If we didn't, the model would try backpropagating all the way to start of the dataset.  
        optimizer.zero_grad()  
          
        outputs = model(task, source, whole_word, source_mask, labels=target)  
        loss = outputs.loss  
        loss.backward()  
        optimizer.step()  
          
        batch_size = task.size(0)  
        text_loss += batch_size * loss.item()  
        total_sample += batch_size  
          
        if (step + 1) % args.log_interval == 0 or step == train_iterator.total_step - 1:  
            cur_t_loss = text_loss / total_sample  
            print(now_time() + 'text loss {:4.4f} | {:5d}/{:5d} batches'.format(  
                cur_t_loss, step + 1, train_iterator.total_step))
            
            # 紀錄訓練 loss
            log_loss(epoch, step + 1, cur_t_loss)
            
            text_loss = 0.  
            total_sample = 0  
  
def evaluate(iterator):  
    if iterator is None:  
        return float('inf')  
          
    # Turn on evaluation mode which disables dropout.  
    model.eval()  
    text_loss = 0.  
    total_sample = 0  
      
    with torch.no_grad():  
        for step in range(iterator.total_step):  
            task, source, source_mask, whole_word, target = iterator.next_batch_valid()  
            task = task.to(device)  
            source = source.to(device)  
            source_mask = source_mask.to(device)  
            whole_word = whole_word.to(device)  
            target = target.to(device)  
              
            outputs = model(task, source, whole_word, source_mask, labels=target)  
            loss = outputs.loss  
              
            batch_size = task.size(0)  
            text_loss += batch_size * loss.item()  
            total_sample += batch_size  
              
    return text_loss / total_sample  
  
# 保存初始模型  
with open(model_path, 'wb') as f:  
    torch.save(model, f)  
  
print(now_time() + 'Start training')  
# Loop over epochs.  
best_val_loss = float('inf')  
endure_count = 0  
  
for epoch in range(1, args.epochs + 1):  
    print(now_time() + 'epoch {}'.format(epoch))  
    train(epoch)  
    print(now_time() + 'validation')  
      
    val_loss = evaluate(valid_iterator)  
    print(now_time() + 'validation loss {:4.4f}'.format(val_loss))
    
    # 紀錄驗證 loss
    log_loss(epoch, 0, None, val_loss)
      
    # Save the model if the validation loss is the best we've seen so far.  
    if val_loss < best_val_loss:  
        best_val_loss = val_loss  
        with open(model_path, 'wb') as f:  
            torch.save(model, f)  
        print(now_time() + 'Model saved')  
    else:  
        endure_count += 1  
        print(now_time() + 'Endured {} time(s)'.format(endure_count))  
        if endure_count == args.endure_times:  
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')  
            break  
  
print(now_time() + 'Training completed')
print(now_time() + f'Loss log saved to: {loss_log_path}')