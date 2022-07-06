import fairseq

ckpt_path = "/workspace/models/hubert_model/hubert_base_ls960.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
print(model['final_proj'])

print(model)
print(cfg)
print(task)
