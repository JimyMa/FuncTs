from typing import Dict, List

import torch
import functs

# functs init
def _jit_pass_tensorssa_remove_update(g: torch.Graph) -> None: ...
def _jit_pass_convert_to_tensorssa(g: torch.Graph) -> None: ...
def _jit_pass_dumb_remove_inter_precedure_mutation(g: torch.Graph) -> None: ...
def _jit_pass_remove_inplace(g: torch.Graph) -> None: ...

class TensorSSAMutateInfo:
    mutNodes: Dict[torch.Value, List[torch.Node]]
    mutValues: List[torch.Value]

def _jit_pass_rewrite_mutation(g: torch.Graph, mutateInfo: functs._C.TensorSSAMutateInfo) -> None: ...
def _jit_pass_block_propagation(g: torch.Graph, mutateInfo: functs._C.TensorSSAMutateInfo) -> None: ...
def _jit_pass_rename(g: torch.Graph) -> None: ...


def _jit_pass_fait_shape_infer(g: torch.Graph, type_hint: List[torch.Type]) -> None: ...
def _jit_pass_fait_pipeline(g: torch.Graph, type_hint: List[torch.Type]) -> None: ...

def _jit_pass_freeze(module: torch.ScriptModule) -> None: ...
def _jit_pass_clone(module: torch.ScriptModule) -> None: ...

def _jit_get_code(g: torch.Graph) -> torch._C.Code: ...

# functs script init
def _create_function_from_graph(g: torch.Graph) -> functs._C.ScriptFunction: ...





