import torch.nn as nn
import torch
import torch.nn.functional as F


def transformer_encoder_model(
        embeddings=256,
        heads=8,
        dim_feedforward=512,
        batch_first=True,
        depth=6
    ):
    encoder_layer = nn.TransformerEncoderLayer(
            d_model=embeddings,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            batch_first=batch_first
        )
    transformer_enc = nn.TransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=depth,
        )
    return transformer_enc


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
