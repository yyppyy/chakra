#!/usr/bin/env python3

import logging
from io import TextIOWrapper
from typing import Any, List
import math
import csv
import ast
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

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

def squareish_groups_fast(mesh_x, mesh_y, G):
    # aspect-weighted split
    gx = max(1, int(round(math.sqrt(G * (mesh_x / mesh_y)))))
    gy = math.ceil(G / gx)
    return mesh_x // gx, mesh_y // gy

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
        group_x,
        group_y,
        alltoall_dispatch_send_matrix,
    ) -> None:
        try:
            self.tokens = tokens
            self.hidden = hidden
            self.expert_hidden = expert_hidden
            self.gemm1_comm1 = gemm1_comm1
            self.gemm1_comm2 = gemm1_comm2
            self.gemm2_comm1 = gemm2_comm1
            self.gemm2_comm2 = gemm2_comm2
                    
            len_flat = group_x * group_y
            k_x, k_y = 0, 0

            gemm1_k_para_flat = closest_divisor(len_flat, (hidden * len_flat / expert_hidden) ** (1/2))
            # # decompose k_para_flat into x and y
            # for k_x in range(1, group_x + 1):
            #     if (gemm1_k_para_flat % k_x == 0) and (group_x % k_x == 0):
            #         k_y = gemm1_k_para_flat // k_x
            #         if group_y % k_y == 0:
            #             break
            # assert group_x % k_x == 0 and group_y % k_y == 0
            # self.gemm1_part_x = group_x // k_x
            # self.gemm1_part_y = group_y // k_y
            self.gemm1_part_x, self.gemm1_part_y = squareish_groups_fast(group_x, group_y, gemm1_k_para_flat)

            gemm2_k_para_flat = closest_divisor(len_flat, (expert_hidden * len_flat / hidden) ** (1/2))
            # # decompose k_para_flat into x and y
            # for k_x in range(1, group_x + 1):
            #     if (gemm2_k_para_flat % k_x == 0) and (group_x % k_x == 0):
            #         k_y = gemm2_k_para_flat // k_x
            #         if group_y % k_y == 0:
            #             break
            # assert group_x % k_x == 0 and group_y % k_y == 0
            # self.gemm2_part_x = group_x // k_x
            # self.gemm2_part_y = group_y // k_y
            self.gemm2_part_x, self.gemm2_part_y = squareish_groups_fast(group_x, group_y, gemm2_k_para_flat)

            self.alltoall_dispatch_send_matrix = alltoall_dispatch_send_matrix

            self.mesh_x = mesh_x
            self.mesh_y = mesh_y
            self.group_x = group_x
            self.group_y = group_y

            print(f'mesh({self.mesh_x},{self.mesh_y}), group({self.group_x},{self.group_y}), part1({self.gemm1_part_x},{self.gemm1_part_y}), part2({self.gemm2_part_x},{self.gemm2_part_y})')

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
    
    def get_alltoall_matrix_by_npu(self, npu_id: int):
        alltoall_dispatch_send_matrix = []
        for dst, msg_size in enumerate(self.alltoall_dispatch_send_matrix[npu_id]):
            if dst != npu_id and msg_size > 0:
                alltoall_dispatch_send_matrix += [dst, msg_size]

        alltoall_dispatch_recv_matrix = []
        for src, msg_sizes in enumerate(self.alltoall_dispatch_send_matrix):
            msg_size = msg_sizes[npu_id]
            if src != npu_id and msg_size > 0:
                alltoall_dispatch_recv_matrix += [src, msg_size]
        
        return (alltoall_dispatch_send_matrix, alltoall_dispatch_recv_matrix,
                alltoall_dispatch_recv_matrix, alltoall_dispatch_send_matrix)
    

    def get_group_id_by_npu(self, npu_id: int):
        i = npu_id // self.mesh_y
        j = npu_id % self.mesh_y
        group_i = i // self.group_x
        group_j = j // self.group_y
        return group_i * (self.mesh_y // self.group_y) + group_j


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

    # def get_layers(self, f: TextIOWrapper, num_layers: int) -> List[MoELayer]:
    #     # hardcoded for now
    #     npu_count = 16

    #     layer = MoELayer(
    #         tokens = [1 for _ in range(npu_count)],
    #         hidden = 7168,
    #         expert_hidden = 2048,
    #         gemm1_comm1 = "ALLGATHER",
    #         gemm1_comm2 = None,
    #         gemm2_comm1 = None,
    #         gemm2_comm2 = "REDUCESCATTER",
    #         mesh_x = 4,
    #         mesh_y = 4,
    #         alltoall_dispatch_send_matrix = [[1 if i != j else 0 for i in range(npu_count)]
    #                                                             for j in range(npu_count)],
    #     )

    #     layers = [layer, ]
    #     return layers

    def get_layers(self, token_routing: List[int]) -> List[MoELayer]:
        
        # model and mesh configs hardcoded for now
        npu_count = 1024
        mesh_x, mesh_y = 32, 32
        hidden = 7168
        expert_hidden = 2046

        # get edge npus
        def edge_tile_ids_col_major(mesh_x: int, mesh_y: int):
            """Edge tile ids for a mesh_x × mesh_y grid using column-major flattening:
            id = x*mesh_y + y. Order: top row L→R, right col T→B (no corners),
            bottom row R→L, left col B→T (no corners).
            """
            if mesh_x <= 0 or mesh_y <= 0:
                return []

            tid = lambda x, y: x * mesh_y + y
            ids = []

            # Top edge (y=0)
            for x in range(mesh_x):
                ids.append(tid(x, 0))

            # Right edge (x=mesh_x-1), excluding corners
            if mesh_x > 1 and mesh_y > 2:
                for y in range(1, mesh_y - 1):
                    ids.append(tid(mesh_x - 1, y))
            elif mesh_x > 1 and mesh_y == 2:
                # no middle points to add
                pass

            # Bottom edge (y=mesh_y-1), only if height >= 2
            if mesh_y > 1:
                for x in range(mesh_x - 1, -1, -1):
                    ids.append(tid(x, mesh_y - 1))

            # Left edge (x=0), excluding corners
            if mesh_x > 1 and mesh_y > 2:
                for y in range(mesh_y - 2, 0, -1):
                    ids.append(tid(0, y))

            return ids

        edge_npu_ids = edge_tile_ids_col_major(mesh_x, mesh_y)

        all_npu_ids = list(range(npu_count))

        total_msg_size = sum(token_routing) * hidden
        msg_size = total_msg_size // npu_count

        alltoall_dispatch_send_matrix = [[0 for i in range(npu_count)] for j in range(npu_count)]

        sender_npu_idx = 0
        for recver_npu_id in range(npu_count):
            sender_npu_id = edge_npu_ids[sender_npu_idx]
            if sender_npu_id != recver_npu_id:
                alltoall_dispatch_send_matrix[sender_npu_id][recver_npu_id] += msg_size
            sender_npu_idx += 1
            if sender_npu_idx == len(edge_npu_ids):
                sender_npu_idx = 0

        # todo: gemm is per ep group not global...
        num_groups = len(token_routing)

        group_x, group_y = squareish_groups_fast(mesh_x, mesh_y, num_groups)

        layer = MoELayer(
            tokens = token_routing,
            hidden = hidden,
            expert_hidden = expert_hidden,
            gemm1_comm1 = "ALLGATHER",
            gemm1_comm2 = "REDUCESCATTER",
            gemm2_comm1 = "ALLGATHER",
            gemm2_comm2 = "REDUCESCATTER",
            mesh_x = mesh_x,
            mesh_y = mesh_y,
            group_x = group_x,
            group_y = group_y,
            alltoall_dispatch_send_matrix = alltoall_dispatch_send_matrix,
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

    def get_comp_node(self, name: str, ops: int, data_mov: int) -> Any:
        node = self.get_node("COMP_NODE_" + name, COMP_NODE)
        # node.duration_micros = comp_time
        node.attr.append(ChakraAttr(name="num_ops", int64_val=ops))
        node.attr.append(ChakraAttr(name="tensor_size", int64_val=data_mov)) # assume no register cache, so all weights/tokens need to be load from SRAM every time
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

    def get_comm_coll_node(
        self,
        name: str,
        comm_type: str,
        comm_size: int,
        group_x: int = None,
        group_y: int = None,
        part_x: int = None,
        part_y: int = None,
        inter_part: bool = None,
        alltoall_send_matrix: List = None,
        alltoall_recv_matrix: List = None,) -> Any:

        node = self.get_node(f"COMM_COLL_NODE_{name}_{comm_type}", COMM_COLL_NODE)
        node.attr.append(ChakraAttr(name="comm_type", int64_val=self.get_comm_type(comm_type)))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
        node.attr.append(ChakraAttr(name="group_x", int32_val=group_x))
        node.attr.append(ChakraAttr(name="group_y", int32_val=group_y))
        node.attr.append(ChakraAttr(name="partition_x", int32_val=part_x))
        node.attr.append(ChakraAttr(name="partition_y", int32_val=part_y))
        node.attr.append(ChakraAttr(name="inter_partition", bool_val=inter_part))
        
        if alltoall_send_matrix is not None:
            assert alltoall_recv_matrix is not None
            a = ChakraAttr(name=f"alltoall_send_matrix")
            a.int32_list.values.extend(alltoall_send_matrix)     # packed in proto3
            node.attr.append(a)
            b = ChakraAttr(name=f"alltoall_recv_matrix")
            b.int32_list.values.extend(alltoall_recv_matrix)     # packed in proto3
            node.attr.append(b)

        return node

    def add_parent(self, child_node: Any, parent_node: Any) -> None:
        child_node.data_deps.append(parent_node.id)

    def convert(self) -> None:

        # Columns defining a unique configuration combo
        COMBO_COLS = [
            "Layer",
            "SRAM Capacity Factor",
            "Batch Per Chip",
            "Num Expert Groups",
            "Expert Grouping",
            "Expert Group Placement",
            "TP Algo.",
            "Token Routing Algo.",
        ]

        REQ_COL = "Request ID"
        ROUTING_COL = "Token Routing"


        def load_token_routing_by_combo(csv_path: Path) -> Dict[
            Tuple, Dict[int, List[List[int]]]
        ]:
            """
            Return:
            {
                (combo tuple): { request_id: token_routing_list_of_lists, ... },
                ...
            }
            """
            df = pd.read_csv(csv_path)

            # Parse "Token Routing" strings into Python lists safely
            def parse_routing(s: str):
                if pd.isna(s):
                    return []
                s = s.strip()
                # Some rows may be unquoted list (e.g., [[9]]), some quoted.
                # ast.literal_eval can handle both.
                try:
                    val = ast.literal_eval(s)
                except Exception:
                    # Last-ditch: wrap in brackets if it's a flat number
                    try:
                        val = [[int(s)]]
                    except Exception:
                        val = []
                return val

            df[ROUTING_COL] = df[ROUTING_COL].apply(parse_routing)

            result: Dict[Tuple, Dict[int, List[List[int]]]] = {}

            for combo_vals, grp in df.groupby(COMBO_COLS, dropna=False):
                # Ensure deterministic order by Request ID
                grp = grp.sort_values(REQ_COL)
                token_routings = [
                    ast.literal_eval(s) if isinstance(s, str) else (s if isinstance(s, list) else [])
                    for s in grp[ROUTING_COL].tolist()
                ]
                result[combo_vals] = token_routings

            return result

        # combos is a dict of list that maps each combo to list of token routing
        combos = load_token_routing_by_combo(self.input_filename)

        max_combo = 1
        max_batch = 1

        combo_cnt = 0
        for combo in combos:
            # print(combo, combos[combo])
            # return
            for batch_id, token_routing in enumerate(combos[combo][:max_batch]):
                self.convert_model_parallel(
                    batch_id,
                    '_'.join(str(_) for _ in combo),
                    [sum(t) for t in token_routing]
                )
            
            combo_cnt += 1
            if combo_cnt == max_combo:
                break

            # first_line = f.readline().strip().split()
            # parallelism_type = first_line[0]
            # num_layers = int(f.readline().strip())

            # if parallelism_type == "MICRO":
            #     self.convert_microbenchmark(f, num_layers)
            # elif parallelism_type == "DATA":
            #     self.convert_data_parallel(f, num_layers)
            # elif parallelism_type == "MODEL":
            #     self.convert_model_parallel(f, num_layers)
            # elif parallelism_type == "HYBRID_DATA_MODEL":
            #     self.convert_hybrid_data_model(f, num_layers)
            # elif parallelism_type == "HYBRID_MODEL_DATA":
            #     self.convert_hybrid_model_data(f, num_layers)
            # elif (parallelism_type == "HYBRID_DLRM") or (parallelism_type == "HYBRID_DLRM_ENHANCED"):
            #     last_bottom_layer = int(first_line[1])
            #     self.convert_hybrid_dlrm(f, num_layers, last_bottom_layer)
            # else:
            #     raise ValueError(f"Unsupported parallelism type, {parallelism_type}")

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

    def convert_model_parallel(self, batch_id, combo, token_routing) -> None:

        layers = self.get_layers(token_routing)

        for npu_id in range(self.num_npus):

            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)

                # no attention for now
                assert len(layers) == 1

                # forward pass
                for idx, layer in enumerate(layers):
                    
                    alltoall_dispatch_send_matrix, alltoall_dispatch_recv_matrix,\
                    alltoall_combine_send_matrix, alltoall_combine_recv_matrix = layer.get_alltoall_matrix_by_npu(npu_id)

                    # npu_tokens = layer.tokens[npu_id]
                    # tot_tokens = sum(layer.tokens)
                    # avg_tokens = tot_tokens // self.num_npus
                    last_node = None
                    group_id = layer.get_group_id_by_npu(npu_id)

                    layer.dispatch_comm_node = self.get_comm_coll_node(
                        f'Layer{idx}_DISPATCH',
                        'ALLTOALL',
                        sum(layer.tokens) * layer.hidden,
                        1,
                        1,
                        1,
                        1,
                        True,
                        alltoall_dispatch_send_matrix,
                        alltoall_dispatch_recv_matrix)
                    last_node = layer.dispatch_comm_node
                    encode_message(g, layer.dispatch_comm_node)

                    if layer.gemm1_comm1 is not None:
                        layer.gemm1_comm1_node = self.get_comm_coll_node(
                            f'Layer{idx}_GEMM1_COMM1',
                            layer.gemm1_comm1,
                            layer.tokens[group_id] * layer.hidden,
                            layer.group_x,
                            layer.group_y,
                            layer.gemm1_part_x,
                            layer.gemm1_part_y,
                            False)
                        if last_node is not None:
                            self.add_parent(layer.gemm1_comm1_node, last_node)
                        last_node = layer.gemm1_comm1_node
                        encode_message(g, layer.gemm1_comm1_node)
                    
                    ops = layer.tokens[group_id] * layer.hidden * layer.expert_hidden // (layer.group_y * layer.group_y)
                    layer.gemm1_comp_node = self.get_comp_node(
                        f'Layer{idx}_GEMM1',
                        ops,
                        ops * 3, # 3 due to 3 operands per GEMM op
                    )
                    if last_node is not None:
                        self.add_parent(layer.gemm1_comp_node, last_node)
                    last_node = layer.gemm1_comp_node
                    encode_message(g, layer.gemm1_comp_node)
                    
                    if layer.gemm1_comm2 is not None:
                        layer.gemm1_comm2_node = self.get_comm_coll_node(
                            f'Layer{idx}_GEMM1_COMM2',
                            layer.gemm1_comm2,
                            layer.tokens[group_id] * layer.hidden,
                            layer.group_x,
                            layer.group_y,
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
                            layer.tokens[group_id] * layer.hidden,
                            layer.group_x,
                            layer.group_y,
                            layer.gemm2_part_x,
                            layer.gemm2_part_y,
                            False)
                        if last_node is not None:
                            self.add_parent(layer.gemm2_comm1_node, last_node)
                        last_node = layer.gemm2_comm1_node
                        encode_message(g, layer.gemm2_comm1_node)
                    
                    ops = layer.tokens[group_id] * layer.hidden * layer.expert_hidden // (layer.group_y * layer.group_y)
                    layer.gemm2_comp_node = self.get_comp_node(
                        f'Layer{idx}_GEMM2',
                        ops,
                        ops * 3,
                    )
                    if last_node is not None:
                        self.add_parent(layer.gemm2_comp_node, last_node)
                    last_node = layer.gemm2_comp_node
                    encode_message(g, layer.gemm2_comp_node)
                    
                    if layer.gemm2_comm2 is not None:
                        layer.gemm2_comm2_node = self.get_comm_coll_node(
                            f'Layer{idx}_GEMM2_COMM2',
                            layer.gemm2_comm2,
                            layer.tokens[group_id] * layer.hidden,
                            layer.group_x,
                            layer.group_y,
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
                        sum(layer.tokens) * layer.hidden,
                        1,
                        1,
                        1,
                        1,
                        True,
                        alltoall_combine_send_matrix,
                        alltoall_combine_recv_matrix)
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
