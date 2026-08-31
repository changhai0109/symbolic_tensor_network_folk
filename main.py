import os
import argparse
import sympy as sp
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.grad_updater import (
    GradUpdater,
    MicroBatchReplicator,
    MicroBatchReplicatorPostProcess,
)
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.graph_distributer import GraphDistributer
from symbolic_tensor_graph.graph.convert_chakra import BundledConvertChakra
import re
from symbolic_tensor_graph.vram_counting import _print_gpu_vram

mixprecision = False


def str_to_bool(v):
    return v.lower() in ("true", "t", "1", "yes", "y")


def _create_pipeline_tensor_map_mix_precision(
    _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks
):
    _tensor_map = dict()
    assert len(_temporal_parallel_dims) == 1
    parallel_dim = _temporal_parallel_dims[0]
    range_ = _symbol_map_value[parallel_dim]

    num_stacks_each_stage = [num_stacks // range_] * range_
    for i in range(num_stacks % range_):
        num_stacks_each_stage[i] += 1
    cumulative = []
    acc = 0
    for v in num_stacks_each_stage:
        acc += v
        cumulative.append(acc)

    for tensor in _tensors:
        tid = tensor.id
        m = re.search(r"transformer\.(\d+)", tid)
        if m:
            block_idx = int(m.group(1))
            stage = next(i for i, up in enumerate(cumulative) if block_idx < up)
            _tensor_map[tid] = {parallel_dim: stage}
            continue
        if "in_emb" in tid:
            _tensor_map[tid] = {parallel_dim: 0}
        elif "out_emb" in tid or "loss" in tid:
            _tensor_map[tid] = {parallel_dim: (range_ - 1)}
        else:
            raise ValueError(f"Unrecognized tensor id for pipeline mapping: {tid}")

    return _tensor_map


def _create_pipeline_tensor_map(
    _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks
):
    if mixprecision:
        return _create_pipeline_tensor_map_mix_precision(
            _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks
        )
    _tensor_map = dict()
    assert len(_temporal_parallel_dims) == 1
    parallel_dim = _temporal_parallel_dims[0]
    range_ = _symbol_map_value[parallel_dim]
    num_stacks_each_stage = list()
    for i in range(range_):
        num_stacks_each_stage.append(num_stacks // range_)
    for i in range(num_stacks - range_ * (num_stacks // range_)):
        num_stacks_each_stage[i] += 1
    for i in range(range_):
        if i == 0:
            continue
        num_stacks_each_stage[i] += num_stacks_each_stage[i - 1]

    for tensor in _tensors:
        if tensor.id == "transformer.18._sharded_weight@1":
            pass
        found = False
        for num_stack in range(num_stacks):
            if f"transformer.{num_stack}." in tensor.id:
                for stage, upper_bound in enumerate(num_stacks_each_stage):
                    if num_stack < upper_bound:
                        _tensor_map[tensor.id] = {parallel_dim: stage}
                        found = True
                        break
                if found:
                    break
        if found:
            pass
        elif "in_emb" in tensor.id:
            _tensor_map[tensor.id] = {parallel_dim: 0}
        elif "out_emb" in tensor.id:
            _tensor_map[tensor.id] = {parallel_dim: (num_stacks - 1) % range_}
        elif "loss" in tensor.id:
            _tensor_map[tensor.id] = {parallel_dim: (num_stacks - 1) % range_}
        else:
            assert False, tensor.name
    return _tensor_map


def _apply_microbatch_replication(graph, symbol_map_value):
    """Apply microbatch replication with env-based optimization toggle."""
    if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") == "0":
        return MicroBatchReplicator.apply(graph, symbol_map_value)
    else:
        print("[Warning] MICROBATCH OPTIMIZE sometimes generate incorrect graphs, use with caution!")
        return ReplicateGraph.apply(
            graph,
            inplace=True,
            old_symbol_map_new_symbol={"Batch": "MicroBatch"},
        )


def _apply_weight_sharding(graph, weight_sharded):
    """Apply weight sharding (fsdp) replication."""
    if weight_sharded:
        return ReplicateGraph.apply(
            graph,
            inplace=True,
            old_symbol_map_new_symbol={"fsdp": "dp"},
        )
    else:
        return ReplicateGraph.apply(
            graph, inplace=True, old_symbol_map_new_symbol={"fsdp": 1}
        )


def _get_readout_backend():
    """Return the backend class for chakra readout.

    Default is the Chakra v0.0.4 (protobuf) backend. Set
    ``STAGE_READOUT_BACKEND=json`` to fall back to the JSON backend.
    """
    if os.environ.get("STAGE_READOUT_BACKEND", "chakra004") == "json":
        from symbolic_tensor_graph.chakra.backends.json_backend import JsonBackend
        return JsonBackend
    from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
        Chakra004Backend,
    )
    return Chakra004Backend


def _distribute_convert_readout(graph, symbol_map_value, spatial_parallel_dims,
                                temporal_parallel_dims, pipeline_tensor_map,
                                args, generated_filename, label):
    """Run the full pipeline: GraphDistributer -> BundledConvertChakra -> readout."""
    print(f"{label} model: Distributing")
    distributed_graph = GraphDistributer.apply(
        graph,
        symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        pipeline_tensor_map,
    )

    if args.print_gpu_vram:
        _print_gpu_vram(
            distributed_graph,
            symbol_map_value,
            mixed_precision=args.mixed_precision,
            header=f"[{label}] ",
        )

    print(f"{label} model: Converting Chakra")
    comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
    distributed_chakra_graph = BundledConvertChakra.apply(
        distributed_graph,
        symbol_map_value,
        os.path.join(args.output_dir, comm_group_file),
        mixed_precision=args.mixed_precision,
    )

    print(f"{label} model: reading out")
    if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") != "0":
        distributed_chakra_graph = MicroBatchReplicatorPostProcess.apply(
            distributed_chakra_graph, args.batch // args.micro_batch
        )
    distributed_chakra_graph.readout(generated_filename, backend=_get_readout_backend())


def _build_parse_map_value(args):
    """Parse CLI args into symbol_map_value and sympy symbol references."""
    os.makedirs(args.output_dir, exist_ok=True)
    if "%d" not in args.output_name:
        backend = os.environ.get("STAGE_READOUT_BACKEND", "chakra004")
        ext = ".json" if backend == "json" else ".et"
        args.output_name = f"{args.output_name}.%d{ext}"
    generated_filename = os.path.join(args.output_dir, args.output_name)

    dp, tp, pp, spp, ep, fsdp = sp.symbols("dp tp pp cp ep fsdp")
    (
        Din, Dout, Dmodel, Dff, Batch, Seq,
        Head, KVHead, Experts, KExperts, Dvocal, MicroBatch,
    ) = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head KVHead Experts KExperts Dvocal MicroBatch"
    )

    if args.micro_batch == -1:
        args.micro_batch = args.batch

    symbol_map_value = {
        Dvocal: args.dvocal,
        Dmodel: args.dmodel,
        Dff: args.dff,
        Batch: args.batch,
        MicroBatch: args.micro_batch,
        Seq: args.seq,
        Head: args.head,
        KVHead: args.kvhead,
        Experts: args.experts,
        KExperts: args.kexperts,
        dp: args.dp,
        tp: args.tp,
        pp: args.pp,
        spp: args.sp,
        ep: args.ep,
    }

    if args.weight_sharded:
        symbol_map_value[fsdp] = args.dp if args.dp != 0 else 1
        symbol_map_value["fsdp"] = args.dp if args.dp != 0 else 1
    else:
        symbol_map_value[fsdp] = 1
        symbol_map_value["fsdp"] = 1

    symbols = (dp, tp, pp, spp, ep, fsdp)
    return generated_filename, symbols, symbol_map_value


def _process_model(graph, symbol_map_value, spatial_parallel_dims,
                   temporal_parallel_dims, num_stacks, args,
                   generated_filename, label, *, absorb_ep_into_tp=False):
    """Shared model processing: tensor map, distribute, convert, readout."""
    if absorb_ep_into_tp:
        symbol_map_value[spatial_parallel_dims[1]] *= symbol_map_value[spatial_parallel_dims[-1]]

    pipeline_tensor_map = _create_pipeline_tensor_map(
        graph.tensors, temporal_parallel_dims, symbol_map_value, num_stacks
    )

    _distribute_convert_readout(
        graph, symbol_map_value, spatial_parallel_dims,
        temporal_parallel_dims, pipeline_tensor_map,
        args, generated_filename, label,
    )


def _build_dense_graph(model_builder, num_stacks, tpsp, symbol_map_value, args):
    """Build a dense/gpt graph with shared wiring."""
    print("Assembling dense model")
    graph = model_builder(num_stacks, regenerate=True, tpsp=tpsp)
    graph = _apply_microbatch_replication(graph, symbol_map_value)
    graph = _apply_weight_sharding(graph, args.weight_sharded)
    graph = GradUpdater.apply(graph, inplace=True)
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, help="dir where stores output traces", required=True
    )
    parser.add_argument(
        "--output_name", type=str, help="name of output traces", required=True
    )
    parser.add_argument("--dp", type=int, help="data parallel degree", required=False, default=1)
    parser.add_argument("--tp", type=int, help="tensor parallel degree", required=False, default=1)
    parser.add_argument("--sp", type=int, help="sequence parallel degree", required=False, default=1)
    parser.add_argument("--ep", type=int, help="expert parallel degree", required=False, default=1)
    parser.add_argument("--pp", type=int, default=1, help="pipeline parallel degree", required=False)
    parser.add_argument(
        "--weight_sharded",
        type=str_to_bool,
        help="whether weight sharded",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--activation_recompute",
        type=str_to_bool,
        help="whether recompute activation",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--tpsp",
        type=str_to_bool,
        help="use tp+sp or tp only",
        required=False,
        default=True,
    )
    parser.add_argument("--dvocal", type=int, default=32000, required=False)
    parser.add_argument("--dmodel", type=int, default=8192, required=False)
    parser.add_argument("--dff", type=int, default=28672, required=False)
    parser.add_argument("--batch", type=int, default=64, required=False)
    parser.add_argument("--micro_batch", type=int, default=-1, required=False)
    parser.add_argument("--seq", type=int, default=1024, required=False)
    parser.add_argument("--head", type=int, default=64, required=False)
    parser.add_argument("--kvhead", type=int, default=8, required=False)
    parser.add_argument("--num_stacks", type=int, default=80, required=False)
    parser.add_argument("--experts", type=int, default=8, required=False)
    parser.add_argument("--kexperts", type=int, default=2, required=False)
    parser.add_argument("--chakra_schema_version", type=str, default="v0.0.4", required=False)
    parser.add_argument("--model_type", type=str, default="dense", required=False)
    parser.add_argument("--mixed_precision", type=str_to_bool, default=False, required=False)
    parser.add_argument(
        "--print_gpu_vram",
        type=str_to_bool,
        default=False,
        required=False,
        help="Whether to print per-GPU VRAM footprint (total / params / acts / grads) in GiB",
    )

    args = parser.parse_args()
    generated_filename, symbols, symbol_map_value = _build_parse_map_value(args)
    dp, tp, pp, spp, ep, fsdp = symbols
    num_stacks = args.num_stacks
    temporal_parallel_dims = [pp]

    global mixprecision
    if args.mixed_precision:
        mixprecision = True

    if args.model_type == "llama" or args.model_type == "dense":
        if mixprecision:
            from models.stage1.llama_model import llama as transformer_fn
        else:
            from models.stage1.gpt_model import gpt as transformer_fn

        graph = _build_dense_graph(transformer_fn, num_stacks, args.tpsp, symbol_map_value, args)
        _process_model(graph, symbol_map_value, [dp, tp, spp],
                       temporal_parallel_dims, num_stacks, args,
                       generated_filename, "Dense", absorb_ep_into_tp=True)

    elif args.model_type == "gpt":
        from models.stage1.gpt_model import gpt as transformer_fn

        graph = _build_dense_graph(transformer_fn, num_stacks, args.tpsp, symbol_map_value, args)
        _process_model(graph, symbol_map_value, [dp, tp, spp],
                       temporal_parallel_dims, num_stacks, args,
                       generated_filename, "Dense", absorb_ep_into_tp=True)

    elif args.model_type == "prefilling":
        from models.stage1.prefilling_model import prefilling as transformer_fn

        print("Assembling prefilling model (dense/llama)")
        graph = transformer_fn(num_stacks, template_dir="tpsp_prefilling", regenerate=True)
        _process_model(graph, symbol_map_value, [dp, tp, spp],
                       temporal_parallel_dims, num_stacks, args,
                       generated_filename, "Prefilling", absorb_ep_into_tp=True)

    elif args.model_type == "prefilling_gpt":
        from models.stage1.prefilling_model import prefilling as transformer_fn

        template_dir = "tpsp_gpt_prefilling" if args.tpsp else "tp_gpt_prefilling"
        print(f"Assembling prefilling GPT model (tpsp={args.tpsp})")
        graph = transformer_fn(num_stacks, template_dir=template_dir, regenerate=True)
        _process_model(graph, symbol_map_value, [dp, tp, spp],
                       temporal_parallel_dims, num_stacks, args,
                       generated_filename, "Prefilling", absorb_ep_into_tp=True)

    elif args.model_type == "prefilling_moe":
        from models.stage1.prefilling_moe_model import prefilling_moe as transformer_fn

        assert args.tpsp
        print("Assembling prefilling MoE model")
        graph = transformer_fn(num_stacks, symbol_map_value, regenerate=True)
        _process_model(graph, symbol_map_value, [dp, tp, spp, ep],
                       temporal_parallel_dims, num_stacks, args,
                       generated_filename, "PrefillingMoE")

    elif args.model_type == "decoding":
        from models.stage1.decoding_model import decoding as transformer_fn

        assert args.sp == 1, "Decoding model requires sp=1 (seq=1 per step, cannot sequence-parallel a single token)"
        print("Assembling decoding model (dense/llama)")
        graph = transformer_fn(num_stacks, template_dir="tpsp_decoding", regenerate=True)
        _process_model(graph, symbol_map_value, [dp, tp, spp],
                       temporal_parallel_dims, num_stacks, args,
                       generated_filename, "Decoding", absorb_ep_into_tp=True)

    elif args.model_type == "decoding_gpt":
        from models.stage1.decoding_model import decoding as transformer_fn

        assert args.sp == 1, "Decoding model requires sp=1 (seq=1 per step, cannot sequence-parallel a single token)"
        template_dir = "tpsp_gpt_decoding" if args.tpsp else "tp_gpt_decoding"
        print(f"Assembling decoding GPT model (tpsp={args.tpsp})")
        graph = transformer_fn(num_stacks, template_dir=template_dir, regenerate=True)
        _process_model(graph, symbol_map_value, [dp, tp, spp],
                       temporal_parallel_dims, num_stacks, args,
                       generated_filename, "Decoding", absorb_ep_into_tp=True)

    elif args.model_type == "decoding_moe":
        from models.stage1.decoding_moe_model import decoding_moe as transformer_fn

        assert args.sp == 1, "Decoding model requires sp=1 (seq=1 per step, cannot sequence-parallel a single token)"
        assert args.tpsp
        print("Assembling decoding MoE model")
        graph = transformer_fn(num_stacks, symbol_map_value, regenerate=True)
        _process_model(graph, symbol_map_value, [dp, tp, spp, ep],
                       temporal_parallel_dims, num_stacks, args,
                       generated_filename, "DecodingMoE")

    elif args.model_type == "moe":
        from models.stage1.moe_model import transformer as transformer_moe

        assert args.tpsp
        print("Assembling moe model")
        graph = transformer_moe(num_stacks, symbol_map_value, regenerate=True)
        graph = _apply_microbatch_replication(graph, symbol_map_value)
        graph = _apply_weight_sharding(graph, args.weight_sharded)
        graph = GradUpdater.apply(graph, inplace=True)

        _process_model(graph, symbol_map_value, [dp, tp, spp, ep],
                       temporal_parallel_dims, num_stacks, args,
                       generated_filename, "MoE")

    elif args.model_type == "debug":
        assert args.pp == 1
        graph = TensorGraph.load_tensor_graph(
            "./sharding_spreadsheets/module3/tpsp/embedding.csv"
        )
        graph = ReplicateGraph.apply(
            graph,
            inplace=True,
            old_symbol_map_new_symbol={
                "Batch": "MicroBatch",
                "Din": "Dvocal",
                "Dout": "Dvocal",
            },
        )
        graph = _apply_weight_sharding(graph, args.weight_sharded)
        graph = GradUpdater.apply(graph, inplace=True)

        pipeline_tensor_map = {
            "x@0": {pp: 0}, "w@0": {pp: 0}, "y@0": {pp: 0},
            "dy@0": {pp: 0}, "dw@0": {pp: 0}, "dx@0": {pp: 0},
            "w@1": {pp: 0},
        }

        _distribute_convert_readout(
            graph, symbol_map_value, [dp, tp, spp, ep],
            temporal_parallel_dims, pipeline_tensor_map,
            args, generated_filename, "MoE",
        )


if __name__ == "__main__":
    main()
