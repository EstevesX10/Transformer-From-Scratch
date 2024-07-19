import pandas as pd
import torch
from Model import (Transformer)
from Validation import (Greedy_Decode)
import altair as alt

def Load_Next_Batch(config:dict, model:Transformer, test_dataloader, vocabulary_source, vocabulary_target, device):
    # load a sample batch from the Test / Validation Set
    batch = next(iter(test_dataloader))
    encoder_input = batch['encoder_input'].to(device)
    encoder_mask = batch['encoder_mask'].to(device)
    decoder_input = batch['decoder_input'].to(device)
    decoder_mask = batch['decoder_mask'].to(device)

    # Convert the Batch into Tokens using the Tokenizer
    encoder_input_tokens = [vocabulary_source.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [vocabulary_target.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

    # Making sure that the batch size == 1
    assert encoder_input.size(0) == 1, "Batch SIze must be equal to 1 to perform Testing / Validation!"
    
    # Perform Inference
    model_output = Greedy_Decode(model, encoder_input, encoder_mask, vocabulary_source, vocabulary_target, config['sequence_length'], device)

    return batch, encoder_input_tokens, decoder_input_tokens

def mtx2df(m:torch.Tensor, max_row:int, max_col:int, row_tokens:list, col_tokens:list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )

def Get_Attention_Map(model:Transformer, attention_type:str, layer:int, head:int) -> torch.Tensor:
    if attention_type == "encoder":
        attention = model.encoder.layers[layer].self_attention_block.attention_scores
    elif attention_type == "decoder":
        attention = model.decoder.layers[layer].self_attention_block.attention_scores
    elif attention_type == "encoder-decoder":
        attention = model.decoder.layers[layer].cross_attention_block.attention_scores
    return attention[0, head].data

def Attention_Map(model:Transformer, attention_type:str, layer:int, head:int, row_tokens:list, col_tokens:list, max_sentence_length:int) -> alt.Chart:
    df = mtx2df(
        Get_Attention_Map(model, attention_type, layer, head),
        max_sentence_length,
        max_sentence_length,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        #.title(f"Layer {layer} Head {head}")
        .properties(height=400, width=400, title=f"Layer {layer} Head {head}")
        .interactive()
    )

def Get_All_Attention_Maps(model:Transformer, attention_type:str, layers:list[int], heads:list[int], row_tokens:list, col_tokens:list, max_sentence_length:int) -> alt.vconcat:
    charts = []
    for layer in layers:
        rowCharts = []
        for head in heads:
            rowCharts.append(Attention_Map(model, attention_type, layer, head, row_tokens, col_tokens, max_sentence_length))
        charts.append(alt.hconcat(*rowCharts))
    return alt.vconcat(*charts)