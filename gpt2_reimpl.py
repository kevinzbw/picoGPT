import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp = np.exp(x-np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x = (x-mean) / np.sqrt(var + eps)
    return x * g + b


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    x = gelu(linear(x, **c_fc))

    # project back down
    x = linear(x, **c_proj)
    return x


# [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
def attention(q, k, v, mask):
    k_t = np.transpose(k, [0, 2, 1])
    return softmax(q @ k_t / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    qkv = linear(x, **c_attn)
    # split into qkv
    # 3*[n_seq, n_embd]
    qkv = np.split(qkv, 3, axis=-1)

    # split into heads
    n_seq, n_embd = x.shape
    qkv = [np.reshape(v, [n_seq, n_head, n_embd//n_head]) for v in qkv]
    # 3*[n_head, n_seq, n_embd]
    q, k, v = [np.transpose(v, [1, 0, 2]) for v in qkv]

    # causal mask to hide future inputs from being attended to
    mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    att = attention(q, k, v, mask[None, :])

    # merge heads
    # [n_seq, n_head, n_embd]
    att = np.transpose(att, [1, 0, 2])
    att = np.reshape(att, [n_seq, n_embd])

    # out projection
    att = linear(att, **c_proj)

    return att


# [n_seq, n_embd] -> [n_seq, n_embd]
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


# [n_seq] -> [n_seq, n_vocab]
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)

    # projection to vocab
    # [n_seq, n_embd] -> [n_seq, n_vocab]
    x = layer_norm(x, **ln_f)
    return x @ wte.T


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    # only return generated ids
    return inputs[len(inputs) - n_tokens_to_generate:]


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(
        model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(
        input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)
