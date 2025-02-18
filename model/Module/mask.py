import torch

def tril_mask(data):#triangular lower matrix
    "Mask out future positions."
    size = data.size(-1) #size为序列长度
    full = torch.full((1,size,size),1,dtype=torch.int,device=data.device)
    mask = torch.tril(full).bool() 
    return mask

#设置对<PAD>的注意力为0
def pad_mask(data, pad=0):
    "Mask out pad positions."
    mask = (data!=pad).unsqueeze(-2)#data!=pad的值设置为true,等于pad设置为false
    return mask 


#计算一个batch数据的src_mask和tgt_mask
class MaskedBatch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = pad_mask(src,pad)
        if tgt is not None:
            self.tgt = tgt[:,:-1] #训练时,拿tgt的每一个词输入,去预测下一个词,所以最后一个词无需输入
            self.tgt_y = tgt[:, 1:] #第一个总是<SOS>无需预测，预测从第二个词开始
            self.tgt_mask = \
                self.make_tgt_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y!= pad).sum() 
    
    @staticmethod
    def make_tgt_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_pad_mask = pad_mask(tgt,pad)#pad
        tgt_tril_mask = tril_mask(tgt)#未来词
        tgt_mask = tgt_pad_mask & (tgt_tril_mask)
        return tgt_mask
    #只有 True & True 的部分才能保留。
    #False 的部分（<PAD> 或未来词）被屏蔽。
    
# 测试tril_mask 
mask = tril_mask(torch.zeros(1,10)) #序列长度为10
print(mask)
print(mask.shape)
#sns.heatmap(mask[0],cmap=sns.cm.rocket);

data = torch.tensor([[1, 2, 0, 0], 
                     [3, 4, 5, 0]])  # batch_size=2, seq_len=4
mask = pad_mask(data, pad=0)

print(mask)

