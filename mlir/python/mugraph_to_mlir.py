#!/usr/bin/env python3
"""
muGraph to MLIR Converter

Converts YiRage muGraph (kernel graph) to YiRage MLIR dialect representation.

Usage:
    from yirage.kernel.graph import KNGraph
    from mlir.python.mugraph_to_mlir import MuGraphToMLIR

    # Create and populate graph
    graph = KNGraph(...)

    # Convert to MLIR
    converter = MuGraphToMLIR()
    mlir_str = converter.convert(graph)

    # Write to file or pass to yirage-opt
    with open('output.mlir', 'w') as f:
        f.write(mlir_str)
"""

from typing import Dict, List, Optional, Tuple, Any
import io


class MuGraphToMLIR:
    """Converts muGraph to MLIR text representation using YiRage dialect."""

    # Map from muGraph operation types to MLIR operations
    OP_MAPPING = {
        # Input/Output
        'kn_input_op': 'input',
        'kn_output_op': 'output',

        # Matrix operations
        'kn_matmul_op': 'yirage.matmul',

        # Unary element-wise operations
        'kn_exp_op': 'math.exp',
        'kn_square_op': 'yirage.square',
        'kn_sqrt_op': 'math.sqrt',
        'kn_silu_op': 'yirage.silu',
        'kn_sigmoid_op': 'yirage.sigmoid',
        'kn_gelu_op': 'yirage.gelu',
        'kn_relu_op': 'yirage.relu',
        'kn_clamp_op': 'yirage.clamp',
        'kn_log_op': 'math.log',

        # Binary element-wise operations
        'kn_add_op': 'arith.addf',
        'kn_mul_op': 'arith.mulf',
        'kn_div_op': 'arith.divf',
        'kn_pow_op': 'math.powf',

        # Reduction operations
        'kn_reduction_0_op': 'yirage.reduce_sum',
        'kn_reduction_1_op': 'yirage.reduce_sum',
        'kn_reduction_2_op': 'yirage.reduce_sum',

        # Normalization
        'kn_rms_norm_op': 'yirage.rms_norm',

        # Concat/Split
        'kn_concat_0_op': 'yirage.concat',
        'kn_concat_1_op': 'yirage.concat',
        'kn_concat_2_op': 'yirage.concat',
        'kn_split_0_op': 'yirage.split',
        'kn_split_1_op': 'yirage.split',
        'kn_split_2_op': 'yirage.split',

        # Communication
        'kn_allreduce_op': 'yirage.allreduce',

        # Customized
        'kn_customized_op': 'yirage.custom',
    }

    # Map from dtype to MLIR type
    DTYPE_CODE_MAPPING = {
        940: "fp16",
        941: "bf16",
        942: "fp32",
    }

    DTYPE_MAPPING = {
        'fp16': 'f16',
        'bf16': 'bf16',
        'fp32': 'f32',
        'fp64': 'f64',
        'int8': 'i8',
        'int16': 'i16',
        'int32': 'i32',
        'int64': 'i64',
        'uint8': 'ui8',
        'uint16': 'ui16',
        'uint32': 'ui32',
        'uint64': 'ui64',
    }

    def __init__(self):
        self.value_counter = 0
        self.tensor_to_ssa: Dict[int, str] = {}  # tensor guid/id -> SSA value name
        self.tensor_types: Dict[int, str] = {}
        self.output = io.StringIO()
        self.indent_level = 0

    def _indent(self) -> str:
        return "  " * self.indent_level

    def _new_value(self) -> str:
        name = f"%{self.value_counter}"
        self.value_counter += 1
        return name

    def _normalize_dtype(self, dtype: Any) -> str:
        if isinstance(dtype, int):
            return self.DTYPE_CODE_MAPPING.get(dtype, "fp16")
        return str(dtype)

    def _get_mlir_type(self, dims: Tuple, dtype: str) -> str:
        """Convert shape and dtype to MLIR tensor type."""
        mlir_dtype = self.DTYPE_MAPPING.get(self._normalize_dtype(dtype), 'f16')
        if dims:
            shape_str = 'x'.join(str(d) for d in dims)
            return f"tensor<{shape_str}x{mlir_dtype}>"
        return f"tensor<*x{mlir_dtype}>"

    def _emit(self, text: str):
        """Emit a line of MLIR."""
        self.output.write(self._indent() + text + "\n")

    def _tensor_key(self, tensor: Any) -> Optional[int]:
        """Stable key for cygraph tensor dicts or dtensor objects."""
        if tensor is None:
            return None
        if isinstance(tensor, dict):
            return int(tensor.get("guid", tensor.get("id", 0)))
        guid = getattr(tensor, "guid", None)
        if guid is not None:
            return int(guid)
        return id(tensor)

    def _tensor_meta(self, tensor: Any, graph) -> Tuple[Tuple[int, ...], str]:
        if isinstance(tensor, dict):
            dims = tuple(int(d) for d in tensor.get("dim", tensor.get("dims", (1,))))
            return dims, str(tensor.get("dtype", "fp16"))
        try:
            dims, _ = graph.cygraph.get_input_dtensor_shape_and_stride(tensor)
            return tuple(int(d) for d in dims), str(tensor.dtype)
        except Exception:
            return (1,), "fp16"

    def _op_inputs(self, op_info: Dict[str, Any]) -> List[Any]:
        if "input_tensors" in op_info:
            return list(op_info.get("input_tensors") or [])
        return list(op_info.get("inputs") or [])

    def _op_outputs(self, op_info: Dict[str, Any]) -> List[Any]:
        if "output_tensors" in op_info:
            return list(op_info.get("output_tensors") or [])
        return list(op_info.get("outputs") or [])

    def _bgraph_tb_types(self, bgraph: Dict[str, Any]) -> set:
        return {str(op.get("op_type", "")) for op in bgraph.get("operators", [])}

    def _has_rms_bgraph(self, bgraph: Optional[Dict[str, Any]]) -> bool:
        if not bgraph:
            return False
        types = self._bgraph_tb_types(bgraph)
        return (
            "tb_forloop_accum_red_ld_rms_op" in types
            or "tb_rms_norm_op" in types
        )

    def _is_fused_rms_matmul_bgraph(self, bgraph: Optional[Dict[str, Any]]) -> bool:
        return self._has_rms_bgraph(bgraph) and "tb_matmul_op" in self._bgraph_tb_types(
            bgraph
        )

    def _emit_rms_norm_only(
        self,
        x_ssa: str,
        x_type: str,
        outputs: List[Any],
    ) -> str:
        elem_dtype = "f16"
        last_dim = 1
        if "x" in x_type:
            shape_part = x_type.split("tensor<")[1].split("x")
            if len(shape_part) >= 2:
                last_dim = int(shape_part[-2])
                elem_dtype = shape_part[-1].rstrip(">")
        gamma = self._emit_unit_gamma(last_dim, elem_dtype)
        gamma_type = f"tensor<{last_dim}x{elem_dtype}>"
        normed = self._new_value()
        self._emit(
            f"{normed} = yirage.rms_norm {x_ssa}, {gamma} "
            f"{{epsilon = 1.000000e-06 : f32}} : {x_type}, {gamma_type} -> {x_type}"
        )
        for out in outputs:
            key = self._tensor_key(out)
            if key is not None:
                self.tensor_to_ssa[key] = normed
                self.tensor_types[key] = x_type
        return normed

    def _emit_rms_matmul_pair(
        self,
        x_ssa: str,
        w_ssa: str,
        x_type: str,
        w_type: str,
        out_type: str,
        outputs: List[Any],
    ) -> str:
        """Emit yirage.rms_norm + yirage.matmul and register output tensors."""
        elem_dtype = "f16"
        last_dim = 1
        if "x" in x_type:
            shape_part = x_type.split("tensor<")[1].split("x")
            if len(shape_part) >= 2:
                last_dim = int(shape_part[-2])
                elem_dtype = shape_part[-1].rstrip(">")
        gamma = self._emit_unit_gamma(last_dim, elem_dtype)
        gamma_type = f"tensor<{last_dim}x{elem_dtype}>"
        normed = self._new_value()
        self._emit(
            f"{normed} = yirage.rms_norm {x_ssa}, {gamma} "
            f"{{epsilon = 1.000000e-06 : f32}} : {x_type}, {gamma_type} -> {x_type}"
        )
        result = self._new_value()
        self._emit(
            f"{result} = yirage.matmul {normed}, {w_ssa} "
            f": {x_type}, {w_type} -> {out_type}"
        )
        for out in outputs:
            key = self._tensor_key(out)
            if key is not None:
                self.tensor_to_ssa[key] = result
                self.tensor_types[key] = out_type
        return result

    def _emit_fused_rms_matmul_customized(self, op_info: Dict[str, Any]) -> bool:
        """Lower kn_customized_op bgraph (RMS accum + matmul) to dialect ops."""
        bgraph = op_info.get("bgraph") or op_info.get("customized_graph")
        if not self._has_rms_bgraph(bgraph):
            return False

        inputs = self._op_inputs(op_info)
        outputs = self._op_outputs(op_info)
        if len(inputs) < 1 or not outputs:
            return False

        x_key = self._tensor_key(inputs[0])
        if x_key is None or x_key not in self.tensor_to_ssa:
            return False

        x_ssa = self.tensor_to_ssa[x_key]
        x_type = self.tensor_types.get(x_key, "tensor<*xf16>")

        operators = getattr(self, "_operators", [])
        has_kn_matmul = any(o.get("op_type") == "kn_matmul_op" for o in operators)
        has_tb_matmul = "tb_matmul_op" in self._bgraph_tb_types(bgraph)

        if has_kn_matmul or not has_tb_matmul:
            self._emit_rms_norm_only(x_ssa, x_type, outputs)
            return True

        if len(inputs) < 2:
            return False
        w_key = self._tensor_key(inputs[1])
        if w_key is None or w_key not in self.tensor_to_ssa:
            return False
        w_ssa = self.tensor_to_ssa[w_key]
        w_type = self.tensor_types.get(w_key, "tensor<*xf16>")

        out = outputs[0]
        if isinstance(out, dict):
            raw_dims = out.get("dim", out.get("dims", (1,)))
            out_dims = tuple(int(d) for d in raw_dims if int(d) > 0) or (1,)
            out_dtype = self._normalize_dtype(
                out.get("dtype", out.get("data_type", "fp16"))
            )
        else:
            try:
                out_dims = tuple(out.dim(i) for i in range(out.num_dims))
                out_dtype = str(out.dtype)
            except Exception:
                out_dims = (1,)
                out_dtype = "fp16"
        out_type = self._get_mlir_type(out_dims, str(out_dtype))

        self._emit_rms_matmul_pair(x_ssa, w_ssa, x_type, w_type, out_type, outputs)
        return True

    def _bgraph_tiling_attrs(self, operators: List[Dict[str, Any]]) -> str:
        """Attach bgraph grid/forloop metadata on @mugraph when tiling is present."""
        for op_info in operators:
            if op_info.get("op_type") != "kn_customized_op":
                continue
            bgraph = op_info.get("bgraph") or op_info.get("customized_graph")
            if not bgraph or not self._is_fused_rms_matmul_bgraph(bgraph):
                continue
            grid = bgraph.get("grid_dim", {})
            gx = max(1, int(grid.get("x", 1)))
            gy = max(1, int(grid.get("y", 1)))
            gz = max(1, int(grid.get("z", 1)))
            fk = max(1, int(bgraph.get("forloop_range", 1)))
            if gx <= 1 and gy <= 1 and gz <= 1 and fk <= 1:
                return ""
            return (
                f" attributes {{yirage.grid_m = {gx} : i64, yirage.grid_y = {gy} : i64, "
                f"yirage.grid_z = {gz} : i64, yirage.forloop_k = {fk} : i64}}"
            )
        return ""

    def _emit_unit_gamma(self, last_dim: int, dtype: str) -> str:
        """Emit rank-1 ones tensor as gamma for unit-scale RMSNorm."""
        mlir_dtype = self.DTYPE_MAPPING.get(str(dtype), "f16")
        gamma_type = f"tensor<{last_dim}x{mlir_dtype}>"
        gamma = self._new_value()
        self._emit(
            f"{gamma} = arith.constant dense<1.0> : {gamma_type}"
        )
        return gamma

    def convert(self, graph) -> str:
        """Convert a KNGraph to MLIR text.

        Args:
            graph: A KNGraph instance (from yirage.kernel.graph)

        Returns:
            MLIR text representation as a string
        """
        self.output = io.StringIO()
        self.value_counter = 0
        self.tensor_to_ssa = {}
        self.tensor_types = {}

        # Get graph structure
        try:
            operators = graph.cygraph.get_graph_structure()
        except AttributeError:
            operators = graph.get_graph_structure() if hasattr(graph, 'get_graph_structure') else []
        self._operators = operators

        # Get input tensors
        try:
            input_dtensors = graph.cygraph.get_input_dtensors()
        except AttributeError:
            input_dtensors = []

        # Build function signature
        input_types = []
        input_ssa_values = []

        for idx, dt in enumerate(input_dtensors):
            try:
                dims, strides = graph.cygraph.get_input_dtensor_shape_and_stride(dt)
                dtype = str(dt.dtype)
            except:
                dims = (1,)
                dtype = 'fp16'

            mlir_type = self._get_mlir_type(dims, dtype)
            input_types.append(mlir_type)

            ssa_name = f"%arg{idx}"
            input_ssa_values.append(ssa_name)
            key = self._tensor_key(dt)
            if key is not None:
                self.tensor_to_ssa[key] = ssa_name
                self.tensor_types[key] = mlir_type

        # Determine output types from graph structure
        output_types = []
        for op_info in operators:
            if op_info.get('op_type') == 'kn_output_op':
                for inp in self._op_inputs(op_info):
                    key = self._tensor_key(inp)
                    if key is not None and key in self.tensor_types:
                        output_types.append(self.tensor_types[key])
                    else:
                        dims, dtype = self._tensor_meta(inp, graph)
                        output_types.append(self._get_mlir_type(dims, dtype))

        if not output_types:
            output_types = ['tensor<*xf16>']  # Default fallback

        # Emit module and function
        self._emit("module {")
        self.indent_level += 1

        # Function signature
        input_sig = ", ".join(f"{name}: {typ}" for name, typ in zip(input_ssa_values, input_types))
        output_sig = ", ".join(output_types)
        tiling_attrs = self._bgraph_tiling_attrs(operators)
        self._emit(f"func.func @mugraph({input_sig}) -> ({output_sig}){tiling_attrs} {{")
        self.indent_level += 1

        # Process each operator
        result_values = []
        for op_info in operators:
            op_type = op_info.get('op_type', 'unknown')

            if op_type == 'kn_input_op':
                # Inputs are already function arguments
                continue
            elif op_type == 'kn_output_op':
                for inp in self._op_inputs(op_info):
                    key = self._tensor_key(inp)
                    if key is not None and key in self.tensor_to_ssa:
                        result_values.append(self.tensor_to_ssa[key])
            else:
                # Regular operation
                self._emit_operation(op_info)

        # Return statement
        if result_values:
            return_str = ", ".join(result_values)
            return_types = ", ".join(output_types[:len(result_values)])
            self._emit(f"return {return_str} : {return_types}")
        else:
            # Fallback: return last computed value
            if self.tensor_to_ssa:
                last_val = list(self.tensor_to_ssa.values())[-1]
                self._emit(f"return {last_val} : {output_types[0]}")
            else:
                self._emit("return")

        self.indent_level -= 1
        self._emit("}")
        self.indent_level -= 1
        self._emit("}")

        return self.output.getvalue()

    def _emit_operation(self, op_info: Dict[str, Any]):
        """Emit MLIR for a single operation."""
        op_type = op_info.get('op_type', 'unknown')
        inputs = self._op_inputs(op_info)
        outputs = self._op_outputs(op_info)

        # Get input SSA values
        input_ssa = []
        input_types = []
        for inp in inputs:
            key = self._tensor_key(inp)
            if key is not None and key in self.tensor_to_ssa:
                input_ssa.append(self.tensor_to_ssa[key])
                input_types.append(self.tensor_types.get(key, "tensor<*xf16>"))
            else:
                placeholder = self._new_value()
                input_ssa.append(placeholder)
                input_types.append("tensor<*xf16>")

        # Determine output type
        if outputs:
            out = outputs[0]
            if isinstance(out, dict):
                out_dims = tuple(int(d) for d in out.get("dim", out.get("dims", (1,))))
                out_dtype = out.get("dtype", "fp16")
            else:
                try:
                    out_dims = tuple(out.dim(i) for i in range(out.num_dims))
                    out_dtype = str(out.dtype)
                except Exception:
                    out_dims = (1,)
                    out_dtype = "fp16"
            out_type = self._get_mlir_type(out_dims, out_dtype)
        else:
            out_type = 'tensor<*xf16>'

        result = self._new_value()

        # Emit operation based on type
        if op_type == "kn_customized_op":
            if self._emit_fused_rms_matmul_customized(op_info):
                return

        if op_type == 'kn_matmul_op':
            if len(input_ssa) >= 2:
                lhs_t = input_types[0] if len(input_types) > 0 else out_type
                rhs_t = input_types[1] if len(input_types) > 1 else out_type
                self._emit(
                    f"{result} = yirage.matmul {input_ssa[0]}, {input_ssa[1]} "
                    f": {lhs_t}, {rhs_t} -> {out_type}"
                )
            else:
                self._emit(f"// matmul requires 2 inputs, got {len(input_ssa)}")

        elif op_type in ('kn_silu_op', 'kn_gelu_op', 'kn_relu_op'):
            yirage_op = op_type.replace('kn_', 'yirage.').replace('_op', '')
            if input_ssa:
                self._emit(f"{result} = {yirage_op} {input_ssa[0]} : {out_type}")

        elif op_type == 'kn_rms_norm_op':
            if input_ssa:
                in_type = input_types[0]
                elem_dtype = "fp16"
                last_dim = 1
                if "x" in in_type:
                    shape_part = in_type.split("tensor<")[1].split("x")
                    if len(shape_part) >= 2:
                        last_dim = int(shape_part[-2])
                        elem_dtype = shape_part[-1].rstrip(">")
                gamma = (
                    input_ssa[1]
                    if len(input_ssa) >= 2
                    else self._emit_unit_gamma(last_dim, elem_dtype)
                )
                gamma_type = f"tensor<{last_dim}x{elem_dtype}>"
                self._emit(
                    f"{result} = yirage.rms_norm {input_ssa[0]}, {gamma} "
                    f"{{epsilon = 1.000000e-06 : f32}} : {in_type}, {gamma_type} -> {out_type}"
                )

        elif op_type in ('kn_add_op', 'kn_mul_op', 'kn_div_op'):
            arith_op = self.OP_MAPPING[op_type]
            if len(input_ssa) >= 2:
                self._emit(f"{result} = {arith_op} {input_ssa[0]}, {input_ssa[1]} : {out_type}")

        elif op_type in ('kn_reduction_0_op', 'kn_reduction_1_op', 'kn_reduction_2_op'):
            axis = int(op_type.split('_')[-2])  # Extract axis from op name
            if input_ssa:
                self._emit(f"{result} = yirage.reduce_sum {input_ssa[0]} {{axis = {axis} : i64}} : {out_type}")

        elif op_type in ('kn_exp_op', 'kn_sqrt_op', 'kn_log_op'):
            math_op = self.OP_MAPPING[op_type]
            if input_ssa:
                self._emit(f"{result} = {math_op} {input_ssa[0]} : {out_type}")

        elif 'concat' in op_type:
            axis = int(op_type.split('_')[-2]) if 'concat' in op_type else 0
            inputs_str = ", ".join(input_ssa)
            self._emit(f"{result} = yirage.concat [{inputs_str}] {{axis = {axis} : i64}} : {out_type}")

        elif 'split' in op_type:
            axis = int(op_type.split('_')[-2]) if 'split' in op_type else 0
            if input_ssa:
                self._emit(f"{result} = yirage.split {input_ssa[0]} {{axis = {axis} : i64}} : {out_type}")

        else:
            # Generic fallback
            self._emit(f"// {op_type}: {input_ssa} -> {result}")
            if input_ssa:
                result = input_ssa[0]  # Pass through

        # Register output tensors
        for out in outputs:
            key = self._tensor_key(out)
            if key is not None:
                self.tensor_to_ssa[key] = result
                self.tensor_types[key] = out_type

    def convert_from_json(self, json_path: str) -> str:
        """Convert a muGraph from JSON file to MLIR.

        Args:
            json_path: Path to the JSON file containing muGraph

        Returns:
            MLIR text representation
        """
        import json

        with open(json_path, 'r') as f:
            graph_data = json.load(f)

        return self._convert_json_data(graph_data)

    def _convert_json_data(self, graph_data: Dict) -> str:
        """Convert JSON graph data to MLIR."""
        self.output = io.StringIO()
        self.value_counter = 0
        self.tensor_to_ssa = {}
        self.tensor_types = {}  # Track types for each tensor id

        operators = graph_data.get('operators', [])
        inputs = graph_data.get('inputs', [])
        outputs = graph_data.get('outputs', [])

        # Build input types
        input_types = []
        input_ssa = []
        for idx, inp in enumerate(inputs):
            dims = tuple(inp.get('dims', [1]))
            dtype = inp.get('dtype', 'fp16')
            mlir_type = self._get_mlir_type(dims, dtype)
            input_types.append(mlir_type)
            ssa = f"%arg{idx}"
            input_ssa.append(ssa)
            inp_id = inp.get('id', idx)
            self.tensor_to_ssa[inp_id] = ssa
            self.tensor_types[inp_id] = mlir_type

        # Build output types
        output_types = []
        for out in outputs:
            dims = tuple(out.get('dims', [1]))
            dtype = out.get('dtype', 'fp16')
            output_types.append(self._get_mlir_type(dims, dtype))

        if not output_types:
            output_types = ['tensor<*xf16>']

        # Emit
        self._emit("module {")
        self.indent_level += 1

        input_sig = ", ".join(f"{name}: {typ}" for name, typ in zip(input_ssa, input_types))
        output_sig = ", ".join(output_types)
        self._emit(f"func.func @mugraph({input_sig}) -> ({output_sig}) {{")
        self.indent_level += 1

        for op in operators:
            self._emit_json_operation(op)

        # Return
        result_vals = [self.tensor_to_ssa.get(out.get('id', 0), '%0') for out in outputs]
        if result_vals:
            self._emit(f"return {', '.join(result_vals)} : {', '.join(output_types)}")

        self.indent_level -= 1
        self._emit("}")
        self.indent_level -= 1
        self._emit("}")

        return self.output.getvalue()

    def _emit_json_operation(self, op: Dict):
        """Emit operation from JSON representation."""
        op_type = op.get('type', 'unknown')
        inputs = op.get('inputs', [])
        outputs = op.get('outputs', [])

        input_ssa = [self.tensor_to_ssa.get(i, f'%{i}') for i in inputs]

        # Get input types from stored info
        input_types = [self.tensor_types.get(i, 'tensor<*xf16>') for i in inputs]

        result = self._new_value()

        out_type = 'tensor<*xf16>'
        if outputs:
            out_info = outputs[0] if isinstance(outputs[0], dict) else {'id': outputs[0]}
            dims = tuple(out_info.get('dims', [1]))
            dtype = out_info.get('dtype', 'fp16')
            out_type = self._get_mlir_type(dims, dtype)

        # Map and emit
        if 'matmul' in op_type.lower():
            if len(input_ssa) >= 2:
                self._emit(f"{result} = yirage.matmul {input_ssa[0]}, {input_ssa[1]} : {input_types[0]}, {input_types[1]} -> {out_type}")
        elif 'silu' in op_type.lower():
            self._emit(f"{result} = yirage.silu {input_ssa[0]} : {out_type}")
        elif 'gelu' in op_type.lower():
            self._emit(f"{result} = yirage.gelu {input_ssa[0]} : {out_type}")
        elif 'relu' in op_type.lower():
            self._emit(f"{result} = yirage.relu {input_ssa[0]} : {out_type}")
        elif 'rms_norm' in op_type.lower():
            self._emit(f"{result} = yirage.rms_norm {input_ssa[0]} : {out_type}")
        elif 'add' in op_type.lower():
            if len(input_ssa) >= 2:
                self._emit(f"{result} = arith.addf {input_ssa[0]}, {input_ssa[1]} : {input_types[0]}")
        elif 'mul' in op_type.lower():
            if len(input_ssa) >= 2:
                self._emit(f"{result} = arith.mulf {input_ssa[0]}, {input_ssa[1]} : {input_types[0]}")
        else:
            self._emit(f"// {op_type}")
            result = input_ssa[0] if input_ssa else '%0'

        # Register outputs with their types
        for out in outputs:
            out_id = out.get('id', out) if isinstance(out, dict) else out
            self.tensor_to_ssa[out_id] = result
            self.tensor_types[out_id] = out_type


def convert_mugraph_to_mlir(graph, output_path: Optional[str] = None) -> str:
    """Convenience function to convert muGraph to MLIR.

    Args:
        graph: KNGraph instance or path to JSON file
        output_path: Optional path to write MLIR output

    Returns:
        MLIR text representation
    """
    converter = MuGraphToMLIR()

    if isinstance(graph, str):
        # Assume it's a JSON path
        mlir_text = converter.convert_from_json(graph)
    else:
        # Assume it's a KNGraph
        mlir_text = converter.convert(graph)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(mlir_text)

    return mlir_text


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mugraph_to_mlir.py <graph.json> [output.mlir]")
        print("\nConverts a muGraph JSON file to MLIR representation.")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    mlir_text = convert_mugraph_to_mlir(input_path, output_path)

    if not output_path:
        print(mlir_text)
