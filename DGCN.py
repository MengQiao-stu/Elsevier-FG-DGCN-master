import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,A:torch.Tensor):
        super(GCNLayer, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1= torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.GCN_liner_theta_1 =nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_theta_2 = nn.Sequential(nn.Linear(input_dim, 2921))
        self.GCN_liner_out_1 =nn.Sequential( nn.Linear(input_dim, output_dim))
        nodes_count=self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask=torch.ceil( self.A*0.00001)
        
    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat
    
    def forward(self, H, model='normal'):

        H = self.BN(H)
        H_xx1 = self.GCN_liner_theta_1(H)

        A1 = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        A2 = F.cosine_similarity(H_xx1, H_xx1, dim=-1)
        A = A1

        if model != 'normal': A=torch.clamp(A,0.1)
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)

        return output,A

class DGCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, model='normal'):
        super(DGCN, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model=model
        self.norm_col_Q = Q / (torch.sum(Q, 1, keepdim=True))
        self.liner_1 = nn.Sequential(nn.Linear(150, 128))
        
        layers_count = 2

        self.GCN_Branch=nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128, self.A))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128, self.A))

        self.Softmax_linear =nn.Sequential(nn.Linear(128, self.class_count))
    
    def forward(self, x: torch.Tensor):
        (h, w, c) = x.shape
        clean_x = self.liner_1(x)
        
        clean_x_flatten=clean_x.reshape([h * w, -1])

        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)

        H = superpixels_flatten
        if self.model=='normal':
            for i in range(len(self.GCN_Branch)): H, _ = self.GCN_Branch[i](H)
        else:
            for i in range(len(self.GCN_Branch)): H, _ = self.GCN_Branch[i](H,model='smoothed')

        GCN_result = torch.matmul(self.Q, H)
        Y = GCN_result
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y

