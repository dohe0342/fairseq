git pull
fairseq-hydra-train \
	--config-dir /workspace/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h_summit \
	common.user_dir=examples/data2vec \
    task.data=/workspace/LibriSpeech/manifests \
    model.w2v_path=/workspace/models/data2vec_model/audio_base_ls.pt \
	+task.uses_branch=False
