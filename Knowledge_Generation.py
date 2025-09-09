from einops import rearrange
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
import numpy as np




class dual_attention(nn.Module):
    def __init__(self, batch_size=256, num=21, dim=128):
        super(dual_attention, self).__init__()
        self.num = num
        self.dim = dim
        self.batch_size = batch_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.l2_norm = F.normalize
        self.soft = nn.Softmax(dim=-1)
        self.num_heads = 1

        """ 
        z_img, c_img, attn_img, _, _ = self.cross_img(z_img_pre, prototype_img)
        z_txt, c_txt, attn_txt, _, _ = self.cross_txt(z_txt_pre, prototype_txt)
        
        """
        self.cross_img = cross_attn(dim=dim, num=num, head=self.num_heads)
        self.cross_txt = cross_attn(dim=dim, num=num, head=self.num_heads)
        self.proj_c = nn.Identity()
        self.proj_z = nn.Linear(dim, dim)
        self.prototype_token_1 = nn.Linear(dim, num, bias=False)
        self.prototype_img = self.prototype_token_1.weight
        self.prototype_token_2 = nn.Linear(dim, num, bias=False)
        self.prototype_txt = self.prototype_token_2.weight
        self.projector_rep = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        self.projector_prototype = nn.Identity()
        self.projector1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.projector2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def pre(self, z_img, prototype_img):
        prototype_img = F.normalize(prototype_img, dim=1)
        z_img = F.normalize(z_img, dim=1)
        prototype_img = self.projector_prototype(prototype_img)
        z_img = self.projector_rep(z_img)
        return z_img, prototype_img



    def eval_z(self, z_img, z_txt, index_swap, epoch):
        z_img_pre, prototype_img, z_txt_pre, prototype_txt = self.pre(
            z_img, self.prototype_img, z_txt, self.prototype_txt)
        z_img, c_img, attn_img, _, _ = self.cross_img(z_img_pre, prototype_img)
        z_txt, c_txt, attn_txt, _, _ = self.cross_txt(z_txt_pre, prototype_txt)
        z_img = F.normalize(z_img, dim=1) + F.normalize(z_img_pre, dim=1)
        z_txt = F.normalize(z_txt, dim=1) + F.normalize(z_txt_pre, dim=1)
        z_img = self.proj_z(z_img)
        z_txt = self.proj_z(z_txt)
        z_img = self.l2_norm(z_img, dim=1)
        z_txt = self.l2_norm(z_txt, dim=1)

        return z_img, z_txt, c_img, c_txt, attn_img, attn_txt

    def train_z(self, z_img):
        z_img_pre, prototype_img = self.pre(z_img, self.prototype_img)
        z_img, c_img, attn_img, _, _ = self.cross_img(z_img_pre, prototype_img)
        z_img = F.normalize(z_img, dim=1) + F.normalize(z_img_pre, dim=1)
        z_img = self.proj_z(z_img)
        c_img = F.normalize(c_img, dim=1) + F.normalize(prototype_img, dim=1)
        c_img = self.proj_c(c_img)

        return z_img, c_img, attn_img




    def forward(self, z_img, index_swap):
        """
        Loss
        """
        z_img, c_img, attn_img = self.train_z(z_img)



        attn_img = attn_img.mean(0)

        return z_img, c_img, attn_img


"""
注意力损失
"""
def attention_reg(c_i, c_j):
    p_i = c_i.sum(0).view(-1)
    p_i /= p_i.sum()
    p_j = c_j.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
    ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
    loss_ne = (ne_i + ne_j)/2

    ne_i_onehot = -torch.sum(c_i * torch.log(c_i + 1e-8), dim=1).mean()
    ne_j_onehot = -torch.sum(c_j * torch.log(c_j + 1e-8), dim=1).mean()
    loss_onehot = (ne_i_onehot + ne_j_onehot)*0.02/2
    return loss_ne + loss_onehot


class cross_attn(nn.Module):
    def __init__(self, dim=128, num=10, head=1):
        super(cross_attn, self).__init__()
        self.scale = dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj1 = nn.Linear(dim, dim, bias=True)
        self.v_proj2 = nn.Linear(dim, dim, bias=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.num_heads = head
        self.dim = dim
        self.num = num
        self.proj_z = nn.Identity()
        self.proj_c = nn.Identity()

    def forward(self, z, prototype):
        B = z.size(0)
        q = rearrange(
            self.q_proj(self.norm1(z)),
            "n (h c)-> h n c",
            h=self.num_heads,
            n=B,
            c=self.dim // self.num_heads,
        )
        k = rearrange(
            self.k_proj(self.norm2(prototype)),
            "n (h c)-> h n c",
            n=self.num,
            h=self.num_heads,
            c=self.dim // self.num_heads,
        )
        v = rearrange(
            self.v_proj1(self.norm2(prototype)),
            "n (h c)-> h n c",
            n=self.num,
            h=self.num_heads,
            c=self.dim // self.num_heads,
        )
        v1 = rearrange(
            self.v_proj2(self.norm1(z)),
            "n (h c)-> h n c",
            h=self.num_heads,
            n=B,
            c=self.dim // self.num_heads,
        )
        sim = (q @ k.transpose(-2, -1)) * self.scale
        attn = sim.softmax(dim=-1)
        z_c = rearrange(
            attn @ v,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=B,
            c=self.dim // self.num_heads,
        )
        c_z = rearrange(
            attn.transpose(-2, -1) @ v1,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=self.num,
            c=self.dim // self.num_heads,
        )
        z_c = self.proj_z(z_c)
        c_z = self.proj_c(c_z)
        return z_c, c_z, attn, v, z


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels
