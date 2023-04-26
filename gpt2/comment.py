import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW

# 定义一些超参数
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 5e-5

# 加载预训练的 GPT-2 模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 将模型转移到GPU上（如果有的话）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义一个 Dataset 类来加载和处理数据
class CommentDataset(Dataset):
    def __init__(self, comments, tokenizer):
        self.comments = comments
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        content, comment = self.comments[index]

        # 将输入和输出编码成 token ID 和 attention mask
        input_encoding = self.tokenizer(content, return_tensors='pt', padding=True, truncation=True)
        comment_encoding = self.tokenizer(comment, return_tensors='pt', padding=True, truncation=True)

        # 将评论的 token ID 移到下一个位置，并在序列末尾添加 EOS 标记
        labels = comment_encoding['input_ids'].clone()
        labels[:, :-1] = labels[:, 1:]
        labels[:, -1] = self.tokenizer.eos_token_id

        # 返回编码后的输入和输出
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'comment': comment
        }

# 加载训练数据
train_data = [["内容1", "生成的评论1：评论"], ["内容2", "生成的评论2：评论"]]
train_dataset = CommentDataset(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 定义优化器和损失函数
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练模型
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}")
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 计算模型的输出和损失
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 反向传播和参数更新
        loss.backward()
        optimizer.step()

        # 记录损失值
        total_loss += loss.item()

    print(f"Average loss: {total_loss/len(train_loader):.4f}")
