import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange

class SA(nn.Module):
    def __init__(self, in_c, out_c,dim_head, heads, cls):
        super(SA, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        # self.convH = nn.Conv2d(15, 18, (3, 3), stride=(1, 1), padding=1)
        self.BN_H1 = nn.BatchNorm2d(15)

        self.num_heads = heads
        self.dim_head = dim_head
        self.Hto_q = nn.Linear(15, dim_head * heads, bias=False)
        self.Hto_k = nn.Linear(15, dim_head * heads, bias=False)
        self.Hto_v = nn.Linear(15, dim_head * heads, bias=False)
        self.rescaleH = nn.Parameter(torch.ones(heads, 1, 1))
        # self.rescaleL = nn.Parameter(torch.ones(heads, 1, 1))
        self.projH = nn.Linear(dim_head * heads, 15, bias=True)
        self.LN_H2 = nn.LayerNorm(15)
        # self.Linear = nn.Linear(15, 15, bias=True)


    def forward(self, F_H):
        """
        F_H,F_L: [b,c,h,w]
        """
        #feature embedding
        # f1 = self.convH(F_H)

        F_H = F.relu(self.BN_H1(F_H))

        #stage 1 for feature cross
        b, c, h, w = F_H.shape
        F_H = F_H.permute(0, 2, 3, 1)
        F_H = F_H.reshape(b, h * w, c)

        # qkv的生成
        Hq_inp = self.Hto_q(F_H)
        Hk_inp = self.Hto_k(F_H)
        Hv_inp = self.Hto_v(F_H)
        Hq, Hk, Hv = map(lambda t: rearrange(t, 'b n (d h) -> b h n d', h=self.num_heads),
                      (Hq_inp, Hk_inp, Hv_inp))  # 对qkv调整形状

        # qkv的标准化
        Hq = F.normalize(Hq, dim=-2, p=2) #标准化，倒数第2维除以其2范数,在倒数第2行标准化的原因为是计算c*c的注意力系数，消去了hw(标准化的目的是时期范数相等，减少差异，方便计算)
        Hk = F.normalize(Hk, dim=-2, p=2)

        # 注意力的计算
        attnH = (Hk.transpose(-2, -1) @ Hq)  # A = K^T*Q; attn：d,d !!是不是多头注意力，即 b, heads, d, d（需要验证）!!
        attnH = attnH * self.rescaleH #自适应缩放因子，加速收敛
        attnH = attnH.softmax(dim=-1)


        #特征与注意力相乘
        x_H = Hv @ attnH  # x_H:b,heads,hw,d

        #多头压缩
        x_H = x_H.permute(0, 2, 1, 3)  # x_H:b,hw,heads,d
        x_H = x_H.reshape(b, h * w, self.num_heads * self.dim_head)
        out_H = self.projH(x_H) # out_H:b,hw,c

        #残差连接
        print("mqqqqqq")
        print(F_H.shape)
        print(out_H.shape)
        F_H = F_H + out_H

        #特征 layer normalization
        F_H = F_H.reshape(b, h, w, c)

        F_H = self.LN_H2(F_H)

         # F_H:b,c,h,w

        # F_H = self.Linear(F_H)
        F_H = F_H.permute(0, 3, 1, 2)


        return F_H