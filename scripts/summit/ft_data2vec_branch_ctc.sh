git pull
fairseq-hydra-train \
	--config-dir /workspace/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h \
	common.user_dir=examples/data2vec \
	checkpoint.best_checkpoint_metric=wer_12 \
	dataset.valid_subset=dev-other \
	criterion._name=branch_ctc \
    task.data=/workspace/LibriSpeech/manifests \
    model.w2v_path=/workspace/models/data2vec_model/audio_base_ls.pt \
	+model.branch_ctc_v1=true \
	#+model.overrides=uses_branch:true \
	#task.overrides=uses_branch:true \
