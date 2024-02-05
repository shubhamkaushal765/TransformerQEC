'''
# class MultiSelfAttention(nn.Module)
    Multi-Head Self Attention module for transformer models.
    Args:
    - embeddings (int): Dimensionality of the token embeddings.
    - heads (int): Number of attention heads.
    - mask (bool): Whether to apply masking in self-attention.

# class TransformerBlock(nn.Module)
    Transformer Block module for building transformer models.
    Args:
    - embeddings (int): Dimensionality of the token embeddings.
    - heads (int): Number of attention heads in the multi-head self-attention layer.
    - ff_dimension (int): Dimensionality of the feed-forward layer.
    - mask (bool): Whether to apply masking in the multi-head self-attention layer.

# class Transformer(nn.Module)
    Transformer model for sequence-to-sequence tasks.
    Args:
    - embeddings (int): Dimensionality of the token embeddings.
    - heads (int): Number of attention heads in the multi-head attention layers.
    - depth (int): Number of transformer layers.
    - seq_length (int): Length of the input sequence. If None, it will be determined automatically.
    - num_tokens (int): Number of distinct tokens in the input sequence.
    - output_size (tuple): Output size of the transformer, represented as (channels, height, width).


Source:
https://peterbloem.nl/blog/transformers
https://pypi.org/project/positional-encodings/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encodings.torch_encodings import PositionalEncoding2D


class MultiSelfAttention(nn.Module):
    def __init__(self, embeddings=256, heads=8, mask=False):
        super().__init__()

        assert embeddings % heads == 0, f"Embeddings:{embeddings} and heads:{heads} doesn't fit the architecture."

        self.embeddings = embeddings
        self.heads = heads
        self.mask = mask
        
        self.tokey = nn.Linear(embeddings, embeddings, bias=False)
        self.toquery = nn.Linear(embeddings, embeddings, bias=False)
        self.tovalue = nn.Linear(embeddings, embeddings, bias=False)

        self.output_head = nn.Linear(embeddings, embeddings)


    def forward(self, x):
        """
        x is of shape (batch, seq, embedding)
        """

        # checking the embedding dimension
        b, seq, emb = x.shape
        assert self.embeddings == emb, f"The embedding dimension doesn't match. Model:{self.embeddings}, Input:{emb}."
        
        keys = self.tokey(x)
        queries = self.toquery(x)
        values = self.tovalue(x)

        # these values can now be cut into (# of head) chunks.
        # each chunk will go to seperate SelfAttention.
        # At the end, all chunks will be concatenated to get back the dimensions in the embeddings.
        heads = self.heads
        emb_chunks = emb // heads
        keys = keys.view(b, seq, heads, emb_chunks)
        queries = queries.view(b, seq, heads, emb_chunks)
        values = values.view(b, seq, heads, emb_chunks)

        # fold the heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * heads, seq, emb_chunks)
        queries = queries.transpose(1, 2).contiguous().view(b * heads, seq, emb_chunks)
        values = values.transpose(1, 2).contiguous().view(b * heads, seq, emb_chunks)

        # self-attention, scaling, masking and normalization
        # the output shape is (b * heads, seq, seq)
        attn_weights = torch.bmm(queries, keys.transpose(1, 2))
        attn_weights = attn_weights / (emb ** 0.5)
        if self.mask:
            indices = torch.triu_indices(seq, seq, offset=1)
            attn_weights[:, indices[0], indices[2]] = float("-inf")
        attn_weights = F.softmax(attn_weights, dim=2)

        # applying self-attention to values
        output = torch.bmm(attn_weights, values).view(b, heads, seq, emb_chunks)

        # re-arrange the output to match the input x, and send it through final FFN
        output = output.transpose(1, 2).contiguous().view(b, seq, emb)
        output = self.output_head(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embeddings=256, heads=8, ff_dimension=512, mask=False):

        super().__init__()

        self.embeddings = embeddings
        self.heads = heads
        self.mask = mask

        self.attention = MultiSelfAttention(embeddings, heads, mask)
        self.norm1 = nn.LayerNorm(embeddings)
        self.norm2 = nn.LayerNorm(embeddings)

        self.ff = nn.Sequential(
            nn.Linear(embeddings, ff_dimension),
            nn.ReLU(),
            nn.Linear(ff_dimension, embeddings)
        )


    def forward(self, x):
        """
        x is of shape (batch, seq, embedding)
        """

        # checking the embedding dimension
        b, seq, emb = x.shape
        assert self.embeddings == emb, f"The embedding dimension doesn't match. Model:{self.embeddings}, Input:{emb}."

        attended = self.attention(x)
        x = self.norm1(attended + x)
        feedforward = self.ff(x)
        x = self.norm2(feedforward + x)

        return x


class Transformer(nn.Module):
    """
    Args:
    - embeddings (int): Dimensionality of the token embeddings.
    - heads (int): Number of attention heads in the multi-head attention layers.
    - depth (int): Number of transformer layers.
    - seq_length (int): Length of the input sequence. If None, it will be determined automatically.
    - num_tokens (int): Number of distinct tokens in the input sequence.
    - output_size (tuple): Output size of the transformer, represented as (channels, height, width).
    """
    def __init__(self, embeddings=256, heads=8, depth=6, seq_length=6*6*6, num_tokens=10, output_size=(3, 11, 11)):

        super().__init__()

        self.num_tokens = num_tokens
        self.embeddings = embeddings
        self.output_size = output_size
        self.token_emb = nn.Embedding(num_tokens, embeddings)
        self.pos_emb = PositionalEncoding2D(embeddings)

        tblocks = []
        for _ in range(depth):
            tblocks.append(TransformerBlock(embeddings=embeddings, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        self.to_probs = nn.Linear(embeddings, torch.prod(torch.tensor(output_size)).item())


    def forward(self, x):
        """
        x (input tensor): (batch, xy_coord, round_coord)
        output (output_tensor): (batch, flattened_output_size)
        """

        # embeddings
        position_shape = *(x.shape), self.embeddings
        for_tokens = x.reshape(x.shape[0], -1)
        tokens = self.token_emb(for_tokens)
        # b, seq, k = tokens.shape
        temp_pos = torch.zeros(position_shape)
        positions = self.pos_emb(temp_pos).reshape(x.shape[0], -1, self.embeddings)
        x = tokens + positions

        # transformer blocks
        x = self.tblocks(x)

        # final output layer
        x = x.mean(dim=1) # avg pool
        output = self.to_probs(x)
        # output = F.log_softmax(x, dim=1)
        # output = output.reshape((x.shape[0], *self.output_size))

        return output

if __name__ == '__main__':
    # testing
    input_seq_length = (8, 6, 6, 6)
    x = torch.randint(low=0, high=9, size=(input_seq_length))

    model = Transformer()
    print(model(x))