from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.graph import TensorGraph


_TEMPLATE_BASE = "./sharding_spreadsheets/module3"


def _resolve_template(template_dir, filename):
    return f"{_TEMPLATE_BASE}/{template_dir}/{filename}"


def group_query_attention(template_dir):
    GQA_surrounding_path = _resolve_template(
        template_dir, "group_query_attention_surrounding.csv"
    )
    GQA_kernel_path = _resolve_template(
        template_dir, "group_query_attention_kernel_fused.csv"
    )
    GQA_surrounding = TensorGraph.load_tensor_graph(GQA_surrounding_path)
    GQA_kernel = TensorGraph.load_tensor_graph(GQA_kernel_path)
    GQA_kernel = ReplicateGraph.apply(GQA_kernel, "attn_kernel.%s")
    links = dict()
    links["q"] = "attn_kernel.q"
    links["k_this"] = "attn_kernel.k_this"
    links["v_this"] = "attn_kernel.v_this"
    links["attn_kernel.qkv"] = "attn"
    GQA = ConnectGraph.apply([GQA_surrounding, GQA_kernel], links)
    return GQA


def feed_forward_network(prefilling_dir):
    ffn_path = _resolve_template(prefilling_dir, "llama_feed_forward_network.csv")
    ffn = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(ffn_path),
        "ffn.%s",
        old_symbol_map_new_symbol={"Seq": "1"},
    )
    return ffn


def transformer_decoder_block(template_dir, prefilling_dir):
    layernorm_path = _resolve_template(prefilling_dir, "layer_norm.csv")
    residual_path = _resolve_template(prefilling_dir, "residual.csv")

    input_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path),
        "input_norm.%s",
        old_symbol_map_new_symbol={"Seq": "1"},
    )
    mha = ReplicateGraph.apply(group_query_attention(template_dir), "mha.%s")
    mha_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "mha_res.%s",
        old_symbol_map_new_symbol={"Seq": "1"},
    )
    post_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path),
        "post_attn_norm.%s",
        old_symbol_map_new_symbol={"Seq": "1"},
    )
    ffn = feed_forward_network(prefilling_dir)
    ffn_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "ffn_res.%s",
        old_symbol_map_new_symbol={"Seq": "1"},
    )

    links = dict()
    links["input_norm.y"] = "mha.x"
    links["mha.o"] = "mha_res.x1"
    links["input_norm.x"] = "mha_res.x2"
    links["mha_res.y"] = "post_attn_norm.x"
    links["post_attn_norm.y"] = "ffn.x0"
    links["ffn.xdown"] = "ffn_res.x1"
    links["post_attn_norm.x"] = "ffn_res.x2"

    decoder_block = ConnectGraph.apply(
        [input_layernorm, mha, mha_res, post_layernorm, ffn, ffn_res], links
    )
    return decoder_block


def transformer_decoders(num_layers, decoder_template):
    links = dict()
    decoders = list()
    for i in range(num_layers):
        decoder = ReplicateGraph.apply(decoder_template, f"transformer.{i}.%s")
        decoders.append(decoder)
        if i > 0:
            links[f"transformer.{i-1}.ffn_res.y"] = f"transformer.{i}.input_norm.x"
    decoders = ConnectGraph.apply(decoders, links)
    return decoders


def decoding(num_layers, template_dir="tpsp_decoding", regenerate=False):
    from . import CACHE_DIR
    import os

    prefilling_dir = template_dir.replace("_decoding", "_prefilling")

    cache_filename = os.path.join(
        CACHE_DIR, f"decoding_{template_dir}_{num_layers}.csv"
    )
    if os.path.exists(cache_filename) and not regenerate:
        return TensorGraph.load_tensor_graph(cache_filename)

    embedding_path = _resolve_template(prefilling_dir, "embedding.csv")
    in_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "in_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dvocal", "Dout": "Dmodel", "Seq": "1"},
    )
    out_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "out_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dmodel", "Dout": "Dvocal", "Seq": "1"},
    )

    decoder_template = transformer_decoder_block(template_dir, prefilling_dir)
    decoders = transformer_decoders(num_layers, decoder_template)

    links = dict()
    links["in_emb.y"] = "transformer.0.input_norm.x"
    links[f"transformer.{num_layers-1}.ffn_res.y"] = "out_emb.x"

    transformer = ConnectGraph.apply([decoders, in_emb, out_emb], links)
    transformer.save_tensor_graph(cache_filename)
    return transformer
