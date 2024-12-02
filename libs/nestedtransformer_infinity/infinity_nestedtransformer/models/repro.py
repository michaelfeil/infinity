
import torch

def test_nested(impl=0):
    embed_dim = 768
    num_heads = 12 

    # input data
    hidden_states = torch.randn((2, 8, embed_dim)).to("cuda").to(torch.float16)
    attention_mask = torch.tensor(
        [[False, False, False, False, False, False, True, True],
        [False, False, True, True, True, True, True, True]],
        dtype=torch.bool
    ).to("cuda")

    # layer data
    in_proj_weight = torch.randn((embed_dim * 3, embed_dim)).to("cuda").to(torch.float16)
    in_proj_bias = torch.randn((embed_dim * 3)).to("cuda").to(torch.float16)
    out_proj_weight = torch.randn((embed_dim, embed_dim)).to("cuda").to(torch.float16)
    out_proj_bias = torch.randn((embed_dim)).to("cuda").to(torch.float16)
    use_gelu = True
    norm_first = False
    norm1_eps = 1e-12
    norm1_weight = torch.randn((embed_dim)).to("cuda").to(torch.float16)
    norm1_bias = torch.randn((embed_dim)).to("cuda").to(torch.float16)
    norm2_weight = torch.randn((embed_dim)).to("cuda").to(torch.float16)
    norm2_bias = torch.randn((embed_dim)).to("cuda").to(torch.float16)
    linear1_weight = torch.randn((embed_dim*4, embed_dim)).to("cuda").to(torch.float16)
    linear1_bias = torch.randn((embed_dim*4)).to("cuda").to(torch.float16)
    linear2_weight = torch.randn((embed_dim, embed_dim*4)).to("cuda").to(torch.float16)
    linear2_bias = torch.randn((embed_dim)).to("cuda").to(torch.float16)

    # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
    # 0->false->keep this token -inf->true->mask this token
    
    # nested_tensor with hidden_states[0].shape = (2, 6, 768) and hidden_states[1].shape = (2, 2, 768)
    if impl == 0:
        hidden_states = torch._nested_tensor_from_mask(
            hidden_states, ~attention_mask
        )    
        attention_mask = None
    else:
        hidden_states = torch.nested.masked_select(hidden_states, ~attention_mask.unsqueeze(-1).broadcast_to(hidden_states.shape))
        attention_mask = None


    hidden_states = torch._transformer_encoder_layer_fwd(
        hidden_states,
        embed_dim,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        out_proj_weight,
        out_proj_bias,
        use_gelu,
        norm_first,
        norm1_eps,
        norm1_weight,
        norm1_bias,
        norm2_weight,
        norm2_bias,
        linear1_weight,
        linear1_bias,
        linear2_weight,
        linear2_bias,
        None # removed with nested
    )
    is_last_layer = True
    if hidden_states.is_nested and is_last_layer:
        hidden_states = hidden_states.to_padded_tensor(0.0)
    return hidden_states

if __name__ == "__main__":
    t1 = test_nested(impl=0) # Works
    t2 = test_nested(impl=1) # BREAKS with NotImplementedError: aten._transformer_encoder_layer_fwd.default
    # TODO: compare t1 and t2