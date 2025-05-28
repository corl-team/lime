from functools import partial
from numbers import Number
from typing import Any, List, Optional, Tuple

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table
from numpy import prod
from transformers import PreTrainedModel


def get_shape(val: Any) -> Optional[List[int]]:
    """
    Get the shapes from a jit value object.

    Args:
        val (torch._C.Value): jit value object.

    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():
        return val.type().sizes()
    else:
        return None
      
def zero_flops_func(inputs: List[Any], outputs: List[Any]) -> Number:
  return 0

def output_shape_flops_counter(inputs: List[Any], outputs: List[Any], multiplicator: int = 1) -> Number:
    """
    Count flops for operations maybe including broadcasting.
    """
    outputs_shapes = [get_shape(v) for v in outputs] # calculating outputs shape to take in mind broadcasting ops
    num_elements = prod(outputs_shapes[0])
    return num_elements * multiplicator

def input_shape_flops_counter(inputs: List[Any], outputs: List[Any], multiplicator: int = 1) -> Number:
    """
    Count flops for operations without broadcasting and different output shapes
    """
    inputs_shapes = [get_shape(v) for v in inputs]
    num_elements = prod(inputs_shapes[0])
    return num_elements * multiplicator
  
def sdpa_flops(inputs: List[Any], outputs: List[Any]) -> Number:
    input_shapes = [get_shape(v) for v in inputs]
    B, H, L, D = input_shapes[0]
    total = (4 * B * H * (L**2) * D) + (3 * B * H * (L**2))
    return total

def matmul_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    outputs_shapes = [get_shape(v) for v in outputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = prod(outputs_shapes[0]) * input_shapes[0][-1]
    return flop

@torch.no_grad()
def get_model_flops(model: PreTrainedModel, input_shape: Tuple = (1, 2048)) -> int:
    model.train()
    input_ids = torch.randint(low=0, high=model.config.vocab_size, size=input_shape, device=model.device)
    flops = FlopCountAnalysis(model, input_ids)

    ops_handlers = {
        "aten::embedding": zero_flops_func,
        "aten::vstack": zero_flops_func,
        "aten::silu": output_shape_flops_counter,
        "aten::tanh": output_shape_flops_counter,
        "aten::add": output_shape_flops_counter,
        "aten::mul": output_shape_flops_counter,
        "aten::cos": partial(output_shape_flops_counter, multiplicator=15),
        "aten::sin": partial(output_shape_flops_counter, multiplicator=15),
        "aten::neg": output_shape_flops_counter,
        "aten::pow": partial(output_shape_flops_counter, multiplicator=2),
        "aten::mean": input_shape_flops_counter,
        "aten::sum": input_shape_flops_counter,
        "aten::rsqrt": partial(output_shape_flops_counter, multiplicator=6),
        "aten::scaled_dot_product_attention": sdpa_flops,
        "aten::softmax": output_shape_flops_counter,
        "aten::matmul": matmul_flop_jit,
        "aten::mm": matmul_flop_jit,
    }
    flops.set_op_handle(**ops_handlers)
    
    print(f"Total FLOPs: {flops.total()}")
    print(flop_count_table(flops))
    print(flop_count_str(flops))
    return int(flops.total())