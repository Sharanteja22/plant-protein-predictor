import torch
import esm
import torch.nn as nn


def load_esm_model(device="cpu"):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    return model, alphabet


def load_projection(projection_path, device="cpu"):
    projection = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU()
    )

    projection.load_state_dict(torch.load(projection_path, map_location=device))
    projection = projection.to(device)
    projection.eval()

    return projection


def extract_embeddings_and_attention(
    model,
    alphabet,
    sequence,
    projection,
    device="cpu",
    max_length=1000
):
    if len(sequence) > max_length:
        sequence = sequence[:max_length]

    batch_converter = alphabet.get_batch_converter()
    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        outputs = model(
            tokens,
            repr_layers=[model.num_layers],
            need_head_weights=True
        )

    reps = outputs["representations"][model.num_layers][0, 1:]
    reps_256 = projection(reps)

    attn = outputs["attentions"]
    attn = attn[:, :, :, 1:, 1:]
    attn = attn[:, 0]
    attn = attn[-3:]
    attn = attn.mean(dim=0).mean(dim=0)
    attn = (attn + attn.transpose(0, 1)) / 2.0

    return reps_256.cpu(), attn.cpu()
