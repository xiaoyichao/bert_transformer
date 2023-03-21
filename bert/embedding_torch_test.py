import torch

vocab_size = 21000
embedding_size = 768
seq_len = 64
initializer_range = 0.02

input_ids =  torch.randint(0,vocab_size, (seq_len, embedding_size))

def embedding_lookup(input_ids,vocab_size,embedding_size ):
    print(input_ids.shape)
    if input_ids.ndim == 2:
        input_ids = input_ids.unsqueeze(-1)

    embedding_table = torch.nn.Parameter(
        torch.empty(vocab_size, embedding_size).normal_(mean=0.0, std=initializer_range))

    flat_input_ids = input_ids.view(-1)


    output = embedding_table.index_select(dim=0, index=flat_input_ids)

    input_shape = input_ids.size()

    output = output.view(*input_shape[:-1], -1)


    print(output.shape())

def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            use_position_embeddings=True,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
    """
    Performs various post-processing on a word embedding tensor.

    Args:
        input_tensor: float Tensor of shape [batch_size, seq_length, embedding_size].
        use_token_type: bool. Whether to add embeddings for `token_type_ids`.
        token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            Must be specified if `use_token_type` is True.
        token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
        token_type_embedding_name: string. The name of the embedding table variable
            for token type ids.
        use_position_embeddings: bool. Whether to add position embeddings for the
            position of each token in the sequence.
        position_embedding_name: string. The name of the embedding table variable
            for positional embeddings.
        initializer_range: float. Range of the weight initialization.
        max_position_embeddings: int. Maximum sequence length that might ever be
            used with this model. This can be longer than the sequence length of
            input_tensor, but cannot be shorter.
        dropout_prob: float. Dropout probability applied to the final output tensor.

    Returns:
        float tensor with same shape as `input_tensor`.

    Raises:
        ValueError: One of the tensor shapes or input values is invalid.
    """
    input_shape = input_tensor.size()
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_embeddings = token_type_table(token_type_ids)
        token_type_table = torch.nn.Embedding(token_type_vocab_size, width)
        output += token_type_embeddings

    if use_position_embeddings:
        assert seq_length <= max_position_embeddings
        full_position_embeddings = torch.nn.Embedding(max_position_embeddings, width)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_tensor.device)
        position_embeddings = full_position_embeddings(position_ids)
        position_embeddings = position_embeddings.unsqueeze(0).expand((batch_size, seq_length, width))
        output += position_embeddings

    # output = layer_norm_and_dropout(output, dropout_prob)
    return output
