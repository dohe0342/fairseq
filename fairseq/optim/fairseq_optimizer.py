# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass


class FairseqOptimizer(object):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @classmethod
    def add_args(cls, parser):
        """Add optimizer-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, "_optimizer"):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError("_optimizer must be an instance of torch.optim.Optimizer")
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Reset optimizer instance."""
        if not hasattr(self, "_optimizer"):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError("_optimizer must be an instance of torch.optim.Optimizer")
        self._optimizer = optimizer

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.param_groups:
            for p in param_group["params"]:
                yield p

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def get_lr(self):
        """Return the current learning rate."""
        return self.param_groups[0]["lr"]

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss, retain_graph=False):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        loss.backward(retain_graph=retain_graph)

    def all_reduce_grads(self, module):
        """Manually all-reduce gradients (if required)."""
        if hasattr(module, "all_reduce_grads"):
            module.all_reduce_grads()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    c = c.to(p.grad.device)
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm."""
        return utils.clip_grad_norm_(self.params, max_norm, aggregate_norm_fn)

    def step(self, closure=None, scale=1.0, groups=None):
        """Performs a single optimization step."""
        if self.supports_step_with_scale:
            if self.supports_groups:
                self.optimizer.step(closure, scale=scale, groups=groups)
            else:
                self.optimizer.step(closure, scale=scale)
        else:
            if scale != 1.0:
                self.multiply_grads(1.0 / scale)
            if self.supports_groups:
                self.optimizer.step(closure, groups=groups)
            else:
                self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()

    @property
    def supports_memory_efficient_fp16(self):
        if hasattr(self.optimizer, "supports_memory_efficient_fp16"):
            return self.optimizer.supports_memory_efficient_fp16
        return False

    @property
    def supports_step_with_scale(self):
        if hasattr(self.optimizer, "supports_step_with_scale"):
            return self.optimizer.supports_step_with_scale
        return False

    @property
    def supports_groups(self):
        if hasattr(self.optimizer, "supports_groups"):
            return self.optimizer.supports_groups
        return False

    @property
    def supports_flat_params(self):
        """
        Whether the optimizer supports collapsing of the model
        parameters/gradients into a single contiguous Tensor.
        """
        if hasattr(self.optimizer, "supports_flat_params"):
            return self.optimizer.supports_flat_params
        return False

    def average_params(self):
        pass

    def broadcast_global_state_dict(self, state_dict):
        """
        Broadcasts a global state dict to all ranks.
        Useful for optimizers that shard state between ranks.
        """
        if hasattr(self.optimizer, "broadcast_global_state_dict"):
            return self.optimizer.broadcast_global_state_dict(state_dict)
        else:
            return state_dict

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self.fp32_optimizer.optimizer.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self.fp32_optimizer.zero_grad()
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optimizer.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class LegacyFairseqOptimizer(FairseqOptimizer):
    def __init__(self, args):
        self.args = args
