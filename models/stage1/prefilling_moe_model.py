from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.graph import TensorGraph
from .prefilling_model import group_query_attention, transformer_decoders

TEMPLATE_DIR = "tpsp_moe_prefilling"
_BASE = "./sharding_spreadsheets/module3"


def _resolve(filename):
    return f"{_BASE}/{TEMPLATE_DIR}/{filename}"


def expert_branch():
    ffn_path = _resolve("llama_feed_forward_network.csv")
    moe_wrapper_path = _resolve("expert_wrapper.csv")

    ffn = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(ffn_path),
        "ffn.%s",
        old_symbol_map_new_symbol={"Seq": "Seq*KExperts/(Experts*ep)"},
    )
    moe_wrapper = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(moe_wrapper_path),
        "ldis.%s",
    )

    expert = ConnectGraph.apply(
        [moe_wrapper, ffn],
        {
            "ldis.x_expert": "ffn.x0",
            "ffn.xdown": "ldis.y_expert",
        },
    )
    return expert


def feed_forward_network(symbol_map_value):
    import sympy as sp

    moe_frame_path = _resolve("moe_frame.csv")
    experts, kexperts, ep = sp.symbols("Experts KExperts ep")
    experts = symbol_map_value[experts]
    kexperts = symbol_map_value[kexperts]
    ep = symbol_map_value[ep]
    experts_each_group = experts / ep
    assert experts_each_group == int(experts_each_group)
    experts_each_group = int(experts_each_group)

    expert = expert_branch()
    moe_frame = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(moe_frame_path), "moe.%s"
    )

    links = dict()
    branches = list()

    for i in range(experts_each_group):
        branches.append(ReplicateGraph.apply(expert, f"moe.{i}.%s"))

    moe = ConnectGraph.apply([moe_frame] + branches, links)
    tensor_id_map_tensor = moe.get_tensor_id_map_tensor()

    moe_xrouted = tensor_id_map_tensor["moe.xrouted@0"]
    for i in range(experts_each_group):
        links = dict()
        links["moe.xrouted"] = f"moe.{i}.ldis.x"
        moe = ConnectGraph.apply([moe], links, inplace=True)
        moe.out_tensors.append(moe_xrouted)

    moe.out_tensors.remove(moe_xrouted)

    to_be_reduce_moe_yrouted = list()

    for i in range(experts_each_group):
        branch_ldis_y = tensor_id_map_tensor[f"moe.{i}.ldis.y@0"]
        to_be_reduce_moe_yrouted.append(branch_ldis_y)
        moe.out_tensors.remove(branch_ldis_y)

    from .utils import reduce_chain

    merged_yrouted = reduce_chain(to_be_reduce_moe_yrouted, "moe.yrouted_r%d", amp=0)
    moe.tensors.extend(merged_yrouted)
    if len(merged_yrouted) > 0:
        merged_yrouted[-1].op_attr = "1"
        merged_yrouted_last = merged_yrouted[-1]
    else:
        assert len(to_be_reduce_moe_yrouted) == 1
        merged_yrouted_last = to_be_reduce_moe_yrouted[0]
    moe.out_tensors.append(merged_yrouted_last)

    links = {
        merged_yrouted_last.name: "moe.yrouted",
    }
    moe = ConnectGraph.apply([moe], links)
    return moe


def transformer_decoder_block(symbol_map_value):
    layernorm_path = _resolve("layer_norm.csv")
    residual_path = _resolve("residual.csv")

    input_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path),
        "input_norm.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )
    mha = ReplicateGraph.apply(
        group_query_attention("tpsp_prefilling"), "mha.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )
    mha_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "mha_res.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )
    post_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path),
        "post_attn_norm.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )
    ffn = feed_forward_network(symbol_map_value)
    ffn_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "ffn_res.%s",
        old_symbol_map_new_symbol={"tp": "tp"},
    )

    links = dict()
    links["input_norm.y"] = "mha.x"
    links["mha.o"] = "mha_res.x1"
    links["input_norm.x"] = "mha_res.x2"
    links["mha_res.y"] = "post_attn_norm.x"
    links["post_attn_norm.y"] = "moe.x"
    links["moe.y"] = "ffn_res.x1"
    links["post_attn_norm.x"] = "ffn_res.x2"

    decoder_block = ConnectGraph.apply(
        [input_layernorm, mha, mha_res, post_layernorm, ffn, ffn_res], links
    )
    return decoder_block


def prefilling_moe(num_layers, symbol_map_value, regenerate=False):
    from . import CACHE_DIR
    import sympy as sp
    import os

    experts, kexperts, ep = sp.symbols("Experts KExperts ep")
    experts = symbol_map_value[experts]
    ep = symbol_map_value[ep]
    experts_each_group = experts / ep
    cache_filename = os.path.join(
        CACHE_DIR, f"prefilling_moe_{num_layers}_{int(experts_each_group)}.csv"
    )
    if os.path.exists(cache_filename) and not regenerate:
        return TensorGraph.load_tensor_graph(cache_filename)

    embedding_path = _resolve("embedding.csv")
    in_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "in_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dvocal", "Dout": "Dmodel", "tp": "tp"},
    )
    out_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "out_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dmodel", "Dout": "Dvocal", "tp": "tp"},
    )

    decoder_template = transformer_decoder_block(symbol_map_value)
    decoders = transformer_decoders(num_layers, decoder_template)

    links = dict()
    links["in_emb.y"] = "transformer.0.input_norm.x"
    links[f"transformer.{num_layers-1}.ffn_res.y"] = "out_emb.x"

    transformer = ConnectGraph.apply([decoders, in_emb, out_emb], links)
    transformer.save_tensor_graph(cache_filename)
    return transformer
