#!/usr/bin/env python3

import logging
from io import TextIOWrapper
from typing import Any, List
import math

from ...schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    COMM_COLL_NODE,
    COMP_NODE,
    REDUCE_SCATTER,
    GlobalMetadata,
    Node,
    NodeType,
    AttributeProto as ChakraAttr,
)
from ..third_party.utils.protolib import encodeMessage as encode_message


def closest_divisor(n: int, x: float) -> int:
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    return min(divisors, key=lambda d: abs(d - x))

class MoELayer:
    def __init__(
        self,
        tokens: List[int],
        hidden: int,
        expert_hidden: int,
        gemm1_comm1,
        gemm1_comm2,
        gemm2_comm1,
        gemm2_comm2,
        mesh_x,
        mesh_y,
    ) -> None:
        try:
            self.tokens = tokens
            self.hidden = hidden
            self.expert_hidden = expert_hidden
            self.gemm1_comm1 = gemm1_comm1
            self.gemm1_comm2 = gemm1_comm2
            self.gemm2_comm1 = gemm2_comm1
            self.gemm2_comm2 = gemm2_comm2
                    
            len_flat = mesh_x * mesh_y
            k_x, k_y = 0, 0

            gemm1_k_para_flat = closest_divisor(len_flat, (hidden * len_flat / expert_hidden) ** (1/2))
            # decompose k_para_flat into x and y
            for k_x in range(1, mesh_x + 1):
                if (gemm1_k_para_flat % k_x == 0) and (mesh_x % k_x == 0):
                    k_y = gemm1_k_para_flat // k_x
                    if mesh_y % k_y == 0:
                        break
            assert mesh_x % k_x == 0 and mesh_y % k_y == 0
            self.gemm1_part_x = mesh_x // k_x
            self.gemm1_part_y = mesh_y // k_y

            gemm2_k_para_flat = closest_divisor(len_flat, (expert_hidden * len_flat / hidden) ** (1/2))
            # decompose k_para_flat into x and y
            for k_x in range(1, mesh_x + 1):
                if (gemm2_k_para_flat % k_x == 0) and (mesh_x % k_x == 0):
                    k_y = gemm2_k_para_flat // k_x
                    if mesh_y % k_y == 0:
                        break
            assert mesh_x % k_x == 0 and mesh_y % k_y == 0
            self.gemm2_part_x = mesh_x // k_x
            self.gemm2_part_y = mesh_y // k_y

            # chakra nodes
            self.dispatch_comm_node = None
            self.gemm1_comm1_node = None
            self.gemm1_comp_node = None
            self.gemm1_comm2_node = None
            self.gemm2_comm1_node = None
            self.gemm2_comp_node = None
            self.gemm2_comm2_node = None
            self.combine_comm_node = None
        except Exception:
            raise ValueError(f'Cannot parse the following layer -- "{line}"')


class YamlConverter:
    def __init__(
        self, input_filename: str, output_filename: str, num_npus: int
    ) -> None:
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_npus = num_npus
        self.next_node_id = 0

    def get_global_metadata(self):
        input_text = ""
        with open(self.input_filename, "r") as input_file:
            input_text = input_file.read()
        attr = [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="input_file", string_val=input_text),
        ]
        metadata = GlobalMetadata(attr=attr)
        return metadata

    # def get_layers(self, f: TextIOWrapper, num_layers: int) -> List[Layer]:
    #     layers = []
    #     for line in f:
    #         layers.append(Layer(line))
    #     return layers

    def get_layers(self, f: TextIOWrapper, num_layers: int) -> List[MoELayer]:
        # hardcoded for now
        layer = MoELayer(
            tokens = [1, 1, 1, 1, 1, 1, 1, 1],
            hidden = 7168,
            expert_hidden = 2048,
            gemm1_comm1 = "ALLGATHER",
            gemm1_comm2 = None,
            gemm2_comm1 = None,
            gemm2_comm2 = "REDUCESCATTER",
            mesh_x = 32,
            mesh_y = 32,
        )
        layers = [layer, ]
        return layers

    def get_node(self, name: str, node_type: NodeType) -> Any:
        node = Node()
        node.id = self.next_node_id
        self.next_node_id += 1
        node.name = name
        node.type = node_type
        return node

    def get_comp_node(self, name: str, dim1: int, dim2: int, dim3: int) -> Any:
        node = self.get_node("COMP_NODE_" + name, COMP_NODE)
        # node.duration_micros = comp_time
        node.attr.append(ChakraAttr(name="num_ops", int64_val=dim1*dim2*dim3))
        node.attr.append(ChakraAttr(name="tensor_size", int64_val=dim1*dim2*dim3)) # assume no register cache, so all weights/tokens need to be load from SRAM every time
        return node

    def get_comm_type(self, comm_type: str) -> int:
        if comm_type == "ALLREDUCE":
            return ALL_REDUCE
        elif comm_type == "ALLTOALL":
            return ALL_TO_ALL
        elif comm_type == "ALLGATHER":
            return ALL_GATHER
        elif comm_type == "REDUCESCATTER":
            return REDUCE_SCATTER
        return 0

    def get_comm_coll_node(self, name: str, comm_type: str, comm_size: int, part_x: int, part_y: int, inter_part: bool) -> Any:
        node = self.get_node(f"COMM_COLL_NODE_{name}_{comm_type}", COMM_COLL_NODE)
        node.attr.append(ChakraAttr(name="comm_type", int64_val=self.get_comm_type(comm_type)))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
        node.attr.append(ChakraAttr(name="partition_x", int32_val=part_x))
        node.attr.append(ChakraAttr(name="partition_y", int32_val=part_y))
        node.attr.append(ChakraAttr(name="inter_partition", bool_val=inter_part))
        return node

    def add_parent(self, child_node: Any, parent_node: Any) -> None:
        child_node.data_deps.append(parent_node.id)

    def convert(self) -> None:
        with open(self.input_filename, "r") as f:
            first_line = f.readline().strip().split()
            parallelism_type = first_line[0]
            num_layers = int(f.readline().strip())

            if parallelism_type == "MICRO":
                self.convert_microbenchmark(f, num_layers)
            elif parallelism_type == "DATA":
                self.convert_data_parallel(f, num_layers)
            elif parallelism_type == "MODEL":
                self.convert_model_parallel(f, num_layers)
            elif parallelism_type == "HYBRID_DATA_MODEL":
                self.convert_hybrid_data_model(f, num_layers)
            elif parallelism_type == "HYBRID_MODEL_DATA":
                self.convert_hybrid_model_data(f, num_layers)
            elif (parallelism_type == "HYBRID_DLRM") or (parallelism_type == "HYBRID_DLRM_ENHANCED"):
                last_bottom_layer = int(first_line[1])
                self.convert_hybrid_dlrm(f, num_layers, last_bottom_layer)
            else:
                raise ValueError(f"Unsupported parallelism type, {parallelism_type}")

    def convert_microbenchmark(self, f: TextIOWrapper, num_layers: int) -> None:
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for i in range(self.num_passes):
                    for layer in layers:
                        bwd_wg_comm_node = self.get_comm_coll_node(
                            layer.name, layer.bwd_wg_comm_type, layer.bwd_wg_comm_size
                        )
                        encode_message(g, bwd_wg_comm_node)

    def convert_data_parallel(self, f: TextIOWrapper, num_layers: int) -> None:
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for i in range(self.num_passes):
                    fwd_comp_node = None

                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(layer.name, "FWD", layer.fwd_comp_time)
                        if idx != 0:
                            self.add_parent(fwd_comp_node, layers[idx - 1].fwd_comp_node)
                        if layer.bwd_wg_comm_node is not None:
                            self.add_parent(fwd_comp_node, layer.bwd_wg_comm_node)
                        layer.fwd_comp_node = fwd_comp_node
                        encode_message(g, fwd_comp_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_wg_comp_node = self.get_comp_node(layer.name, "BWD_WG", layer.bwd_wg_comp_time)
                        if idx == 0:
                            if fwd_comp_node is None:
                                raise ValueError("fwd_comp_node is None")
                            self.add_parent(bwd_wg_comp_node, fwd_comp_node)
                        else:
                            self.add_parent(bwd_wg_comp_node, layers[len(layers) - idx].bwd_ig_comp_node)
                        encode_message(g, bwd_wg_comp_node)

                        bwd_wg_comm_node = self.get_comm_coll_node(
                            layer.name, layer.bwd_wg_comm_type, layer.bwd_wg_comm_size
                        )

                        self.add_parent(bwd_wg_comm_node, bwd_wg_comp_node)
                        layer.bwd_wg_comm_node = bwd_wg_comm_node
                        encode_message(g, bwd_wg_comm_node)

                        if idx != (len(layers) - 1):
                            bwd_ig_comp_node = self.get_comp_node(layer.name, "BWD_IG", layer.bwd_ig_comp_time)
                            self.add_parent(bwd_ig_comp_node, bwd_wg_comp_node)
                            layer.bwd_ig_comp_node = bwd_ig_comp_node
                            encode_message(g, bwd_ig_comp_node)

                for layer in layers:
                    layer.bwd_wg_comm_node = None

    def convert_model_parallel(self, f: TextIOWrapper, num_layers: int) -> None:
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)

                # no attention for now
                assert len(layers) == 1

                # forward pass
                for idx, layer in enumerate(layers):
                    npu_tokens = layer.tokens[npu_id]
                    tot_tokens = sum(layer.tokens)
                    avg_tokens = tot_tokens // self.num_npus
                    last_node = None

                    layer.dispatch_comm_node = self.get_comm_coll_node(
                        f'Layer{idx}_DISPATCH',
                        'ALLTOALL',
                        npu_tokens * layer.hidden,
                        1,
                        1,
                        True)
                    last_node = layer.dispatch_comm_node
                    encode_message(g, layer.dispatch_comm_node)

                    if layer.gemm1_comm1 is not None:
                        layer.gemm1_comm1_node = self.get_comm_coll_node(
                            f'Layer{idx}_GEMM1_COMM1',
                            layer.gemm1_comm1,
                            tot_tokens * layer.hidden,
                            layer.gemm1_part_x,
                            layer.gemm1_part_y,
                            False)
                        if last_node is not None:
                            self.add_parent(layer.gemm1_comm1_node, last_node)
                        last_node = layer.gemm1_comm1_node
                        encode_message(g, layer.gemm1_comm1_node)
                    
                    layer.gemm1_comp_node = self.get_comp_node(f'Layer{idx}_GEMM1', avg_tokens, layer.hidden, layer.expert_hidden)
                    if last_node is not None:
                        self.add_parent(layer.gemm1_comp_node, last_node)
                    last_node = layer.gemm1_comp_node
                    encode_message(g, layer.gemm1_comp_node)
                    
                    if layer.gemm1_comm2 is not None:
                        layer.gemm1_comm2_node = self.get_comm_coll_node(
                            f'Layer{idx}_GEMM1_COMM2',
                            layer.gemm1_comm2,
                            tot_tokens * layer.expert_hidden,
                            layer.gemm1_part_x,
                            layer.gemm1_part_y,
                            True)
                        if last_node is not None:
                            self.add_parent(layer.gemm1_comm2_node, last_node)
                        last_node = layer.gemm1_comm2_node
                        encode_message(g, layer.gemm1_comm2_node)
                    
                    if layer.gemm2_comm1 is not None:
                        layer.gemm2_comm1_node = self.get_comm_coll_node(
                            f'Layer{idx}_GEMM2_COMM1',
                            layer.gemm2_comm1,
                            tot_tokens * layer.expert_hidden,
                            layer.gemm2_part_x,
                            layer.gemm2_part_y,
                            False)
                        if last_node is not None:
                            self.add_parent(layer.gemm2_comm1_node, last_node)
                        last_node = layer.gemm2_comm1_node
                        encode_message(g, layer.gemm2_comm1_node)
                    
                    layer.gemm2_comp_node = self.get_comp_node(f'Layer{idx}_GEMM2', avg_tokens, layer.expert_hidden, layer.hidden)
                    if last_node is not None:
                        self.add_parent(layer.gemm2_comp_node, last_node)
                    last_node = layer.gemm2_comp_node
                    encode_message(g, layer.gemm2_comp_node)
                    
                    if layer.gemm2_comm2 is not None:
                        layer.gemm2_comm2_node = self.get_comm_coll_node(
                            f'Layer{idx}_GEMM2_COMM2',
                            layer.gemm2_comm2,
                            tot_tokens * layer.hidden,
                            layer.gemm2_part_x,
                            layer.gemm2_part_y,
                            True)
                        if last_node is not None:
                            self.add_parent(layer.gemm2_comm2_node, last_node)
                        last_node = layer.gemm2_comm2_node
                        encode_message(g, layer.gemm2_comm2_node)

                    layer.combine_comm_node = self.get_comm_coll_node(
                        f'Layer{idx}_COMBINE',
                        'ALLTOALL',
                        npu_tokens * layer.hidden,
                        1,
                        1,
                        True)
                    if last_node is not None:
                            self.add_parent(layer.combine_comm_node, last_node)
                    last_node = layer.combine_comm_node
                    encode_message(g, layer.combine_comm_node)    

    def convert_hybrid_data_model(self, f: TextIOWrapper, num_layers: int) -> None:
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for i in range(self.num_passes):
                    fwd_comm_node = None

                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(layer.name, "FWD", layer.fwd_comp_time)
                        if layer.bwd_wg_comm_node is not None:
                            self.add_parent(fwd_comp_node, layer.bwd_wg_comm_node)
                        if idx != 0:
                            self.add_parent(fwd_comp_node, layers[idx - 1].fwd_comm_node)
                        encode_message(g, fwd_comp_node)

                        fwd_comm_node = self.get_comm_coll_node(layer.name, layer.fwd_comm_type, layer.fwd_comm_size)
                        self.add_parent(fwd_comm_node, fwd_comp_node)
                        layer.fwd_comm_node = fwd_comm_node
                        encode_message(g, fwd_comm_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_ig_comp_node = self.get_comp_node(layer.name, "BWD_IG", layer.bwd_ig_comp_time)
                        if idx == 0:
                            if fwd_comm_node is None:
                                raise ValueError("fwd_comm_node is None")
                            self.add_parent(bwd_ig_comp_node, fwd_comm_node)
                        else:
                            self.add_parent(bwd_ig_comp_node, layers[len(layers) - idx].bwd_wg_comp_node)
                            self.add_parent(bwd_ig_comp_node, layers[len(layers) - idx].bwd_ig_comm_node)
                        encode_message(g, bwd_ig_comp_node)

                        if idx != num_layers - 1:
                            bwd_ig_comm_node = self.get_comm_coll_node(
                                layer.name + "_IG_COMM_", layer.bwd_ig_comm_type, layer.bwd_ig_comm_size
                            )
                            self.add_parent(bwd_ig_comm_node, bwd_ig_comp_node)
                            layer.bwd_ig_comm_node = bwd_ig_comm_node
                            encode_message(g, bwd_ig_comm_node)

                        bwd_wg_comp_node = self.get_comp_node(layer.name, "BWD_WG", layer.bwd_wg_comp_time)
                        self.add_parent(bwd_wg_comp_node, bwd_ig_comp_node)
                        layer.bwd_wg_comp_node = bwd_wg_comp_node
                        encode_message(g, bwd_wg_comp_node)

                        bwd_wg_comm_node = self.get_comm_coll_node(
                            layer.name, layer.bwd_wg_comm_type, layer.bwd_wg_comm_size
                        )
                        self.add_parent(bwd_wg_comm_node, bwd_wg_comp_node)
                        layer.bwd_wg_comm_node = bwd_wg_comm_node
                        encode_message(g, bwd_wg_comm_node)

                for layer in layers:
                    layer.bwd_wg_comm_node = None

    def convert_hybrid_model_data(self, f: TextIOWrapper, num_layers: int) -> None:
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for i in range(self.num_passes):
                    fwd_comm_node = None

                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(layer.name, "FWD", layer.fwd_comp_time)
                        if layer.bwd_wg_comm_node is not None:
                            self.add_parent(fwd_comp_node, layer.bwd_wg_comm_node)
                        if idx != 0:
                            self.add_parent(fwd_comp_node, layers[idx - 1].fwd_comm_node)
                        encode_message(g, fwd_comp_node)

                        fwd_comm_node = self.get_comm_coll_node(layer.name, layer.fwd_comm_type, layer.fwd_comm_size)
                        self.add_parent(fwd_comm_node, fwd_comp_node)
                        layer.fwd_comm_node = fwd_comm_node
                        encode_message(g, fwd_comm_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_ig_comp_node = self.get_comp_node(layer.name, "BWD_IG", layer.bwd_ig_comp_time)
                        if idx == 0:
                            if fwd_comm_node is None:
                                raise ValueError("fwd_comm_node is None")
                            self.add_parent(bwd_ig_comp_node, fwd_comm_node)
                        else:
                            self.add_parent(bwd_ig_comp_node, layers[len(layers) - idx].bwd_wg_comp_node)
                            self.add_parent(bwd_ig_comp_node, layers[len(layers) - idx].bwd_ig_comm_node)
                        encode_message(g, bwd_ig_comp_node)

                        if idx != num_layers - 1:
                            bwd_ig_comm_node = self.get_comm_coll_node(
                                layer.name, layer.bwd_ig_comm_type, layer.bwd_ig_comm_size
                            )
                            self.add_parent(bwd_ig_comm_node, bwd_ig_comp_node)
                            layer.bwd_ig_comm_node = bwd_ig_comm_node
                            encode_message(g, bwd_ig_comm_node)

                        bwd_wg_comp_node = self.get_comp_node(layer.name, "BWD_WG", layer.bwd_wg_comp_time)
                        self.add_parent(bwd_wg_comp_node, bwd_ig_comp_node)
                        layer.bwd_wg_comp_node = bwd_wg_comp_node
                        encode_message(g, bwd_wg_comp_node)

                        bwd_wg_comm_node = self.get_comm_coll_node(
                            layer.name, layer.bwd_wg_comm_type, layer.bwd_wg_comm_size
                        )
                        self.add_parent(bwd_wg_comm_node, bwd_wg_comp_node)
                        layer.bwd_wg_comm_node = bwd_wg_comm_node
                        encode_message(g, bwd_wg_comm_node)

                for layer in layers:
                    layer.bwd_wg_comm_node = None

    def convert_hybrid_dlrm(self, f: TextIOWrapper, num_layers: int, last_bottom_layer: int) -> None:
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for i in range(self.num_passes):
                    fwd_comp_node = None

                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(layer.name, "FWD", layer.fwd_comp_time)
                        if layer.bwd_wg_comm_node is not None:
                            self.add_parent(fwd_comp_node, layer.bwd_wg_comm_node)
                        elif layer.bwd_wg_comp_node is not None:
                            self.add_parent(fwd_comp_node, layer.bwd_wg_comp_node)
                        if idx != 0:
                            self.add_parent(fwd_comp_node, layers[idx - 1].fwd_comp_node)
                        if idx == last_bottom_layer:
                            self.add_parent(fwd_comp_node, layers[0].fwd_comm_node)
                        layer.fwd_comp_node = fwd_comp_node
                        encode_message(g, fwd_comp_node)

                        if layer.fwd_comm_type == "ALLTOALL":
                            fwd_comm_node = self.get_comm_coll_node(
                                layer.name, layer.fwd_comm_type, layer.fwd_comm_size
                            )
                            self.add_parent(fwd_comm_node, fwd_comp_node)
                            layer.fwd_comm_node = fwd_comm_node
                            encode_message(g, fwd_comm_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_wg_comp_node = self.get_comp_node(layer.name, "BWD_WG", layer.bwd_wg_comp_time)
                        if idx == 0:
                            if fwd_comp_node is None:
                                raise ValueError("fwd_comp_node is None")
                            self.add_parent(bwd_wg_comp_node, fwd_comp_node)
                        else:
                            if layers[len(layers) - idx].bwd_ig_comp_node is not None:
                                self.add_parent(bwd_wg_comp_node, layers[len(layers) - idx].bwd_ig_comp_node)
                            if layers[len(layers) - idx - 1].bwd_ig_comm_node is not None:
                                self.add_parent(bwd_wg_comp_node, layers[len(layers) - idx - 1].bwd_ig_comm_node)
                        layer.bwd_wg_comp_node = bwd_wg_comp_node
                        encode_message(g, bwd_wg_comp_node)

                        if layer.bwd_wg_comm_type != "NONE":
                            bwd_wg_comm_node = self.get_comm_coll_node(
                                layer.name, layer.bwd_wg_comm_type, layer.bwd_wg_comm_size
                            )
                            self.add_parent(bwd_wg_comm_node, bwd_wg_comp_node)
                            layer.bwd_wg_comm_node = bwd_wg_comm_node
                            encode_message(g, bwd_wg_comm_node)

                        bwd_ig_comp_node = None
                        if idx != (len(layers) - 1):
                            bwd_ig_comp_node = self.get_comp_node(layer.name, "BWD_IG", layer.bwd_ig_comp_time)
                            self.add_parent(bwd_ig_comp_node, bwd_wg_comp_node)
                            layer.bwd_ig_comp_node = bwd_ig_comp_node
                            encode_message(g, bwd_ig_comp_node)

                        if (len(layers) - idx - 1) == (last_bottom_layer + 1):
                            bwd_ig_comm_node = self.get_comm_coll_node(
                                layers[0].name, layers[0].bwd_ig_comm_type, layers[0].bwd_ig_comm_size
                            )
                            if bwd_ig_comp_node is None:
                                raise ValueError("bwd_ig_comp_node is None")
                            self.add_parent(bwd_ig_comm_node, bwd_ig_comp_node)
                            layers[0].bwd_ig_comm_node = bwd_ig_comm_node
                            encode_message(g, bwd_ig_comm_node)

                for layer in layers:
                    layer.bwd_wg_comm_node = None
                    layer.bwd_wg_comp_node = None
                    layer.bwd_ig_comm_node = None
                    layer.bwd_ig_comp_node = None
