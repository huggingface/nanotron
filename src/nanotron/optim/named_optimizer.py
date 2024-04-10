from typing import Any, Callable, Dict, Iterable, Tuple, Union

import torch

from nanotron.optim.inherit_from_other_optimizer import InheritFromOtherOptimizer


class NamedOptimizer(InheritFromOtherOptimizer):
    """Mimics somewhat the torch optimizer API"""

    def __init__(
        self,
        named_params_or_groups: Iterable[Union[Tuple[str, torch.Tensor], Dict[str, Any]]],
        optimizer_builder: Callable[[Iterable[Dict[str, Any]]], torch.optim.Optimizer],
        weight_decay: float = 0.0,
        param_name_to_lr_scaling = None,
    ):
        assert param_name_to_lr_scaling is not None
        
        id_to_name_decay, id_to_name_no_decay = {}, {}

        # Don't need to check that param_groups are overlapping since the optimizer will do it for me.
        #  https://github.com/pytorch/pytorch/blob/88b3810c94b45f5982df616e2bc4c471d173f491/torch/optim/optimizer.py#L473
        id_to_name_decay.update(
            {id(param): name for name, param in named_params_or_groups["decay"] if id(param) not in id_to_name_decay}
        )
        id_to_name_no_decay.update(
            {
                id(param): name
                for name, param in named_params_or_groups["no_decay"]
                if id(param) not in id_to_name_no_decay
            }
        )

        id_to_name = {**id_to_name_decay, **id_to_name_no_decay}

        name_to_id = {v: k for k, v in id_to_name.items()}
        assert len(id_to_name) == len(name_to_id)

        # TODO(fmom) Pass weight decay value from config here
        params = [
            {"params": [param for _, param in named_params_or_groups["decay"]], "weight_decay": weight_decay},
            {"params": [param for _, param in named_params_or_groups["no_decay"]], "weight_decay": 0.0},
        ]

        # Sanity check
        for param_group in params:
            _params = param_group["params"]
            for param in _params:
                # https://github.com/pytorch/pytorch/issues/100701
                assert param.numel() > 0

        # NOTE: further separate parameters if its name is in model.param_name_to_lr_scaling
        # mup_params = []
        # for param_group in params:
        #     for p in param_group["params"]:
        #         name = id_to_name[id(p)]
        #         non_mup_params = []
                
        #         if name in param_name_to_lr_scaling:
        #             # NOTE: **param_group except "params" 
        #             optim_hyerparameters = {k: v for k, v in param_group.items() if k != 'params'}
        #             mup_params.append({"params": [p], "lr": param_name_to_lr_scaling[name], **optim_hyerparameters})
        #             param_group["params"].remove(p)
        #         else:
        #             non_mup_params.append(p)
        
        num_orig_params = sum([len(param_group["params"]) for param_group in params])
        
        # merged_params = []
        # for param_group in params:
        #     mup_group = {"params": []}
        #     non_mup_group = {"params": []}
            
        #     for p in param_group["params"]:
        #         name = id_to_name[id(p)]
                
        #         if name in param_name_to_lr_scaling:
        #             optim_hyperparameters = {k: v for k, v in param_group.items() if k != 'params'}
        #             mup_group["params"].append(p)
        #             mup_group.update(optim_hyperparameters)
        #             mup_group["lr"] = param_name_to_lr_scaling[name]
        #         else:
        #             non_mup_group["params"].append(p)
            
        #     if mup_group["params"]:
        #         merged_params.append(mup_group)
            
        #     if non_mup_group["params"]:
        #         non_mup_group.update({k: v for k, v in param_group.items() if k != 'params'})
        #         merged_params.append(non_mup_group)
        
        print("merging parameters for mup")
        assert len(param_name_to_lr_scaling) > 0
        print(param_name_to_lr_scaling)

        merged_params = []
        for param_group in params:
            non_mup_params = []
            
            for p in param_group["params"]:
                name = id_to_name[id(p)]
                print(f"param name: {name}")
                
                if name in param_name_to_lr_scaling:
                    mup_group = {k: v for k, v in param_group.items() if k != 'params'}
                    mup_group["params"] = [p]
                    mup_group["lr"] = param_name_to_lr_scaling[name]
                    merged_params.append(mup_group)
                else:
                    non_mup_params.append(p)
            
            if non_mup_params:
                non_mup_group = {k: v for k, v in param_group.items() if k != 'params'}
                non_mup_group["params"] = non_mup_params
                merged_params.append(non_mup_group)
                            
        # NOTE: sanity check that mup_params has the same number of parameters as params
        num_mum_params = sum([len(param_group["params"]) for param_group in merged_params])
        assert num_orig_params == num_mum_params, f"Number of parameters in mup_params ({num_mum_params}) is not equal to the number of parameters in params ({num_orig_params})"
        
        print(f"merged_params has {len(merged_params)} groups")
        super().__init__(optimizer=optimizer_builder(merged_params), id_to_name=id_to_name)

    def state_dict(self) -> dict:
        optim_state_dict = super().state_dict()

        assert "names" not in optim_state_dict

        state_id_to_name = {id(state): self.id_to_name[id(param)] for param, state in self.optimizer.state.items()}
        optim_state_dict["names"] = {
            index: state_id_to_name[id(state)] for index, state in optim_state_dict["state"].items()
        }
        return optim_state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        # TODO @thomasw21: Make a more robust test
        assert set(self.id_to_name.values()) == set(
            state_dict["names"].values()
        ), f"Elements don't match:\n - Elements in `self.id_to_name` that aren't in the other one: {set(self.id_to_name.values()) - set(state_dict['names'].values())}\n - Elements in `state_dict[\"names\"]` that aren't in the other one: {set(state_dict['names'].values()) - set(self.id_to_name.values())}"

        return super().load_state_dict(state_dict)
