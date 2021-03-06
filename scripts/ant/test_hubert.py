import fairseq

ckpt_path = "/workspace/models/hubert_model/hubert_base_ls960.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
for k, v in model.named_parameters():
    if k == 'final_proj':
        del k

print(model)
print(cfg)
print(task)
