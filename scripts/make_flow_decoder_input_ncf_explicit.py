from pathlib import Path
import argparse

import onnx
from onnx import TensorProto, helper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "pretrained" / "CosyVoice-300M"
DEFAULT_ESTIMATOR = MODEL_DIR / "flow.decoder.estimator.fp32.onnx"
DEFAULT_OUTPUT = OUTPUT_DIR / "flow.decoder.estimator.input_ncf_explicit.onnx"
DEFAULT_OUTPUT_4D = OUTPUT_DIR / "flow.decoder.estimator.input_ncf_explicit_4d.onnx"
DEFAULT_OUTPUT_4D_RESHAPE = OUTPUT_DIR / "flow.decoder.estimator.input_ncf_explicit_4d_reshape.onnx"
DEFAULT_OUTPUT_4D_RESHAPE_TIMEFIX = OUTPUT_DIR / "flow.decoder.estimator.input_ncf_explicit_4d_reshape_timefix.onnx"
DEFAULT_OUTPUT_BASELINE_LAYOUTFIX = OUTPUT_DIR / "flow.decoder.estimator.baseline_layoutfix.onnx"


def find_input(model: onnx.ModelProto, input_name: str):
    for value in model.graph.input:
        if value.name == input_name:
            return value
    raise KeyError(input_name)


def rewrite_consumers(model: onnx.ModelProto, source: str, replacement: str) -> None:
    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name == source:
                node.input[idx] = replacement


def append_value_info(model: onnx.ModelProto, name: str, shape: list[int], data_type: int = TensorProto.FLOAT) -> None:
    model.graph.value_info.append(helper.make_tensor_value_info(name, data_type, shape))


def set_input_shape(model: onnx.ModelProto, input_name: str, shape_values: list[int]) -> None:
    value = find_input(model, input_name)
    dims = value.type.tensor_type.shape.dim
    while len(dims) < len(shape_values):
        dims.add()
    while len(dims) > len(shape_values):
        dims.pop()
    for idx, dim_value in enumerate(shape_values):
        dims[idx].dim_value = dim_value


def insert_input_transpose(model: onnx.ModelProto, input_name: str, channels: int) -> None:
    set_input_shape(model, input_name, [2, 500, channels])
    transposed_name = f"{input_name}_ncf_explicit"
    rewrite_consumers(model, input_name, transposed_name)
    node = helper.make_node(
        "Transpose",
        inputs=[input_name],
        outputs=[transposed_name],
        perm=[0, 2, 1],
        name=f"{input_name}.explicit_ncf",
    )
    model.graph.node.insert(0, node)
    append_value_info(model, transposed_name, [2, channels, 500])


def insert_input_4d_normalize(model: onnx.ModelProto, input_name: str, channels: int) -> None:
    set_input_shape(model, input_name, [2, 1, 500, channels])
    squeezed_name = f"{input_name}_btc_3d"
    normalized_name = f"{input_name}_ncf_explicit"
    rewrite_consumers(model, input_name, normalized_name)
    squeeze_node = helper.make_node(
        "Squeeze",
        inputs=[input_name],
        outputs=[squeezed_name],
        axes=[1],
        name=f"{input_name}.squeeze_4d",
    )
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[squeezed_name],
        outputs=[normalized_name],
        perm=[0, 2, 1],
        name=f"{input_name}.explicit_ncf_4d",
    )
    model.graph.node.insert(0, transpose_node)
    model.graph.node.insert(0, squeeze_node)
    append_value_info(model, squeezed_name, [2, 500, channels])
    append_value_info(model, normalized_name, [2, channels, 500])


def insert_input_4d_reshape_only(model: onnx.ModelProto, input_name: str, channels: int) -> None:
    set_input_shape(model, input_name, [2, 1, channels, 500])
    normalized_name = f"{input_name}_ncf_explicit"
    rewrite_consumers(model, input_name, normalized_name)
    reshape_shape_name = f"{input_name}_reshape_shape"
    reshape_shape = helper.make_tensor(
        name=reshape_shape_name,
        data_type=TensorProto.INT64,
        dims=[3],
        vals=[2, channels, 500],
    )
    model.graph.initializer.append(reshape_shape)
    reshape_node = helper.make_node(
        "Reshape",
        inputs=[input_name, reshape_shape_name],
        outputs=[normalized_name],
        name=f"{input_name}.reshape_4d_to_3d",
    )
    model.graph.node.insert(0, reshape_node)
    append_value_info(model, normalized_name, [2, channels, 500])


def replace_time_embeddings_unsqueeze_with_reshape(model: onnx.ModelProto) -> None:
    target_name = "/time_embeddings/Unsqueeze"
    target_index = None
    target_node = None
    for idx, node in enumerate(model.graph.node):
        if node.name == target_name:
            target_index = idx
            target_node = node
            break
    if target_node is None or target_index is None:
        raise KeyError(target_name)
    if target_node.op_type != "Unsqueeze":
        raise ValueError(f"{target_name} is {target_node.op_type}, expected Unsqueeze")
    if len(target_node.input) != 2:
        raise ValueError(f"{target_name} expected 2 inputs, got {len(target_node.input)}")
    reshape_shape_name = "time_embeddings_unsqueeze_shape"
    reshape_shape = helper.make_tensor(
        name=reshape_shape_name,
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[2, 1],
    )
    model.graph.initializer.append(reshape_shape)
    reshape_node = helper.make_node(
        "Reshape",
        inputs=[target_node.input[0], reshape_shape_name],
        outputs=list(target_node.output),
        name=f"{target_name}.reshape_replacement",
    )
    del model.graph.node[target_index]
    model.graph.node.insert(target_index, reshape_node)
    append_value_info(model, target_node.output[0], [2, 1])


def build_transpose_model(model: onnx.ModelProto) -> onnx.ModelProto:
    insert_input_transpose(model, "x", 80)
    insert_input_transpose(model, "mu", 80)
    insert_input_transpose(model, "cond", 80)
    set_input_shape(model, "mask", [2, 500, 1])
    return model


def build_4d_model(model: onnx.ModelProto) -> onnx.ModelProto:
    insert_input_4d_normalize(model, "x", 80)
    insert_input_4d_normalize(model, "mu", 80)
    insert_input_4d_normalize(model, "cond", 80)
    update_mask_for_4d(model)
    return model


def build_4d_reshape_model(model: onnx.ModelProto) -> onnx.ModelProto:
    insert_input_4d_reshape_only(model, "x", 80)
    insert_input_4d_reshape_only(model, "mu", 80)
    insert_input_4d_reshape_only(model, "cond", 80)
    set_input_shape(model, "mask", [2, 1, 500])
    return model


def build_4d_reshape_timefix_model(model: onnx.ModelProto) -> onnx.ModelProto:
    model = build_4d_reshape_model(model)
    replace_time_embeddings_unsqueeze_with_reshape(model)
    return model


def build_baseline_layoutfix_model(model: onnx.ModelProto) -> onnx.ModelProto:
    insert_input_transpose(model, "x", 80)
    insert_input_transpose(model, "mu", 80)
    insert_input_transpose(model, "cond", 80)
    set_input_shape(model, "mask", [2, 500, 1])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_ESTIMATOR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--mode", choices=["transpose", "4d", "4d_reshape", "4d_reshape_timefix", "baseline_layoutfix"], default="transpose")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()

    if args.output == str(DEFAULT_OUTPUT) and args.mode == "4d":
        output_path = DEFAULT_OUTPUT_4D.resolve()
    if args.output == str(DEFAULT_OUTPUT) and args.mode == "4d_reshape":
        output_path = DEFAULT_OUTPUT_4D_RESHAPE.resolve()
    if args.output == str(DEFAULT_OUTPUT) and args.mode == "4d_reshape_timefix":
        output_path = DEFAULT_OUTPUT_4D_RESHAPE_TIMEFIX.resolve()
    if args.output == str(DEFAULT_OUTPUT) and args.mode == "baseline_layoutfix":
        output_path = DEFAULT_OUTPUT_BASELINE_LAYOUTFIX.resolve()

    model = onnx.load(str(input_path))
    if args.mode == "transpose":
        model = build_transpose_model(model)
    elif args.mode == "4d":
        model = build_4d_model(model)
    elif args.mode == "4d_reshape":
        model = build_4d_reshape_model(model)
    elif args.mode == "4d_reshape_timefix":
        model = build_4d_reshape_timefix_model(model)
    else:
        model = build_baseline_layoutfix_model(model)
    onnx.save(model, str(output_path))
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
