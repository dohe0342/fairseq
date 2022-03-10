import os
import ast
import logging

from pathlib import Path

import torch

from fairseq import utils
from fairseq import checkpoint_utils, options, tasks, utils, progress_bar

from examples.speech_recognition.infer import check_args, make_parser, optimize_models, get_dataset_itr

class InspectW2V2(object):
    def __init__(
        self, 
        model,
        tgt_dict,
        criterion,
        use_cuda=True,
        debug_level=False,
    ):
        super().__init__()

        # Model related
        # model.w2v_encoder.w2v_model.feature_grad_mult = 1.0
        # model.w2v_encoder.freeze_finetune_updates = 0
        self.model = model
        # self.tgt_dict = tgt_dict
        self.criterion = criterion

        # self.bos_index = self.tgt_dict.bos_index
        # self.split_index = self.tgt_dict.unk_index + 1
        self.use_cuda = use_cuda

        # Other
        self.debug_level = debug_level
    
    @torch.no_grad()
    def score(
        self,
        sample,
    ):
        assert not self.model.training
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        # Set gradient variables
        model_params_keys = list()
        model_params_vals = list()
        for key, param in self.model.named_parameters():
            if param.grad is None:
                param.requires_grad_()
            model_params_vals.append(param)
            model_params_keys.append(key)

        source = sample["net_input"]["source"]
        source.requires_grad_()

        with torch.autograd.set_grad_enabled(True):
            # Get network results
            # TODO: mask prediction task is performed on pre-training model
            # Thus if we don't have any mask, we cannot calculate infoNCE loss
            net_output = self.model(**sample["net_input"])
            logits = self.model.get_logits(net_output).float()
            target = self.model.get_targets(sample, net_output)

            sample_size = logits.size(0) # Sentence Average

            log_likelihood = logits[torch.arange(0, sample_size), target] # always first channel is target
            log_likelihood = torch.unbind(log_likelihood.float())

            # Calculate gradients
            grads = torch.autograd.grad(
                log_likelihood, # grad needs only scalar value
                model_params_vals,
                allow_unused=True,
                retain_graph=False
            )
        self.model.zero_grad()
        state = {}
        for k, g in zip(model_params_keys, grads):
            # NOTE: mask embedder has no gradients (we set mask to false)
            if g is not None:
                score = torch.square(g)
                mask = torch.logical_or(torch.isinf(score), torch.isnan(score))
                assert mask.float().mean() == 0.0
            else:
                score = None
            state[k] = score

            param.requires_grad_(False)

        return state, sample_size


def platform(args, task=None, model_state=None):
    check_args(args)

    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. Define task
    task = tasks.setup_task(args)

    # 2. Load models and config
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(args.path, separator="\\"),
        arg_overrides=ast.literal_eval(args.model_overrides),
        task=task,
        suffix=args.checkpoint_suffix,
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
        state=model_state,
    )
    logger.info("| loading model(s) from {}".format(args.path))

    # Optimization models
    use_cuda = torch.cuda.is_available() and not args.cpu
    optimize_models(args, use_cuda, models)
    logging.info(models)

    # Load dataset
    task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)

    items = {
        "model": models[0].online_model if hasattr(models[0], "online_model") else models[0],
        "iterator": get_dataset_itr(args, task, models),
        "dictionary": task.target_dictionary,
        "criterion": task.build_criterion(args),
        "use_cuda" : use_cuda,
    }

    return items


if __name__ == "__main__":    
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    
    results_path = Path(args.results_path)
    if results_path is not None and not os.path.exists(results_path):
        os.makedirs(results_path)

    items = platform(args)
    inspector = InspectW2V2(
        items["model"], 
        items["dictionary"], 
        items["criterion"],
        use_cuda=items["use_cuda"]
    )

    ckpt_name = args.path.split("/")[-1].split(".")[0]

    accum_state = {}
    for name, param in items["model"].named_parameters():
        accum_state[name] = 0.0

    accum_weight = 0
    with progress_bar.build_progress_bar(args, items["iterator"]) as t:
        for sample in t:
            sample_id = sample["id"][0].item()
            state, weight = inspector.score(sample)
            for k in accum_state.keys():
                if state[k] is not None:
                    accum_state[k] += state[k]
                else:
                    accum_state[k] = state[k]
            accum_weight += weight

        # accumulate score with total weight
        for k in accum_state.keys():
            if accum_state[k] is not None:
                accum_state[k] = accum_state[k] / accum_weight
            else:
                accum_state[k] = accum_state[k]

        with open(results_path / f"{ckpt_name}_fisher.pt", "wb") as f:
            torch.save(accum_state, f)
