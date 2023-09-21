import torch
import torch.nn as nn
import torch.nn.functional as F
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(2 *embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32))

    def forward(self, x_p,x_g):
        x_p_norm = self.layer_norm1(x_p)
        x_g_norm = self.layer_norm1(x_g)
        query_p = self.query(x_p_norm)
        key_p = self.key(x_p_norm)
        value_p = self.value(x_p_norm)
        query_g = self.query(x_g_norm)
        key_g = self.key(x_g_norm)
        value_g = self.value(x_g_norm)

        attention_scores_pg = torch.matmul(query_p, key_g.transpose(-2, -1))
        attention_probs_pg = torch.nn.functional.softmax(attention_scores_pg, dim=-1)
        Hp = torch.matmul(attention_probs_pg, value_p)

        attention_scores_gp = torch.matmul(query_g, key_p.transpose(-2, -1))
        attention_probs_gp = torch.nn.functional.softmax(attention_scores_gp, dim=-1)
        Hg = torch.matmul(attention_probs_gp, value_g)


        fused_output=torch.cat((Hp+x_p_norm,Hg+x_g_norm),dim=1)
        fused_output=self.layer_norm2(fused_output)
        output=self.mlp(fused_output)
        return output
