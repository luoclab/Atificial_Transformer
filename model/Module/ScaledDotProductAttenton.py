import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    "Compute 'Scaled Dot Product Attention'"
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)  # 获取隐藏维度
        scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)  # 点积注意力计算
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 处理mask
        
        p_attn = F.softmax(scores, dim=-1)  # 归一化注意力得分
        
        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return p_attn @ value, p_attn  # 返回注意力加权值和注意力权重
    
if __name__ =="__main__":

    # 创建查询、键和值向量 (batch_size=1, seq_len=3, d_k=3)
    query = torch.rand(12,512,1024)

    key = torch.rand(12,512,1024)

    value = torch.rand(12,512,1024)
    
    mask = None
    dropout = nn.Dropout(0.1)  # 10% dropout
    # 初始化注意力层
    attention = ScaledDotProductAttention()

    # 计算注意力
    output, attn_weights = attention(query, key, value,mask, dropout)

    # 打印结果
    print("Attention Weights.shape:\n", attn_weights.shape)
    print("Output.shape:\n", output.shape)
