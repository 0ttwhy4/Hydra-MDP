from click import Choice
from cv2 import sqrt
import torch
import torch.nn as nn
import torch.functional as F
import transformer

def multihead_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads_num: int=4):
    assert q.shape == k.shape, f"The query and key should have the same shape, while query shape {q.shape}, key shape {k.shape}"
    bs = q.shape[0]
    seq_len = q.shape[1]
    dim_q = q.shape[2]
    dim_v = v.shape[2]
    assert dim_q % heads_num == 0
    q_head_dim = dim_q // heads_num
    v_head_dim = dim_v // heads_num
    multi_q = q.reshape((bs, seq_len, q_head_dim, -1)).permute(0, 3, 1, 2) # (bs, head_num, seq_len, head_dim)
    multi_k = k.reshape((bs, seq_len, q_head_dim, -1)).permute(0, 3, 1, 2)
    multi_v = v.reshape((bs, seq_len, v_head_dim, -1)).permute(0, 3, 1, 2)
    scale = sqrt(q_head_dim)
    attn_score = torch.einsum('b h q d, b h k d -> b h q k', [multi_q, multi_k]) / scale # (bs, head_num, q_seq_len, k_seq_len)
    attn_score = torch.softmax(attn_score, dim=-1)
    output = torch.einsum('b h q k, b h k v -> b h q v', [attn_score, multi_v])
    return output.permute(0, 2, 1, 3).reshape(output.shape[0], output.shape[1], -1) # (bs, seq_len, embed_dim)

class SelfAttention(nn.Module):

    def __init__(self, embed_dim: int, head_num: int):
        self.project = nn.Linear(embed_dim, 3 * embed_dim)
        self.head_num = head_num

    def forward(self, embeddings):
        qkv = self.project(embeddings) # (bs, seq_len, 3 * embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        return multihead_attention(q, k, v, head_num=self.head_num)
    
class CrossAttention(nn.Module):

    def __init__(self, embed_dim: int, head_num: int):
        self.q_project = nn.Linear(embed_dim, embed_dim)
        self.kv_project = nn.Linear(embed_dim, 2*embed_dim)
        self.head_num = head_num

    def forward(self, q_embedding, kv_embedding):
        q = self.q_project(q_embedding)
        kv = self.kv_project(kv_embedding)
        k, v = kv.chunk(2, dim=-1)
        return multihead_attention(q, k, v, head_num=self.head_num)


class FFN(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):

    def __init__(self, 
                 attn_head_num: int,
                 embedding_size: int):
        self.embedding_size = embedding_size
        self.multihead_attention = SelfAttention(embedding_size, head_num=attn_head_num)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.fc = FFN(embedding_size, embedding_size)

    def forward(self, embeddings):
        embeddings = self.layer_norm1(embeddings + self.multihead_attention(embeddings))
        embeddings = self.layer_norm2(embeddings + self.fc(embeddings))
        return embeddings
    
class DecoderLayer(nn.Module):

    def __init__(self,
                 attn_head_num: int,
                 embedding_size: int,
                 kv_embeddings: torch.Tensor):
        self.embedding_size = embedding_size
        self.attention = CrossAttention(embedding_size, attn_head_num)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.fc = FFN(embedding_size, embedding_size)
        self.kv_embeddings = kv_embeddings

    def forward(self, q_embeddings):
        attn_embeddings = self.attention(q_embeddings, self.kv_embeddings)
        attn_embeddings = self.layer_norm1(attn_embeddings + q_embeddings)
        attn_embeddings = self.layer_norm2(attn_embeddings + self.fc(attn_embeddings))
        return attn_embeddings
    
class Encoder(nn.Module):

    def __init__(self, 
                 num_layers: int=6,
                 attn_head_num: int=8,
                 embedding_size: int=256):
        self.head_num = attn_head_num
        self.embedding_size = embedding_size
        layers = [EncoderLayer(attn_head_num, embedding_size) for _ in range(num_layers)]
        self.encoder = nn.Sequential(*layers)

    def forward(self, embeddings):
        return self.encoder(embeddings)

class Decoder(nn.Module):

    def __init__(self,
                 kv_embeddings: torch.Tensor,
                 num_layers: int=6,
                 attn_head_num: int=8,
                 embedding_size: int=256,):
        self.head_num = attn_head_num
        self.embedding_size = embedding_size
        layers = [DecoderLayer(attn_head_num, embedding_size, kv_embeddings) for _ in range(num_layers)]
        self.decoder = nn.Sequential(*layers)

    def forward(self, q_embeddings):
        return self.decoder(q_embeddings)