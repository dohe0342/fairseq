git pull
fairseq-hydra-train \
	--config-dir /workspace/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h_summit \
	common.user_dir=/workspace/fairseq/examples/data2vec \
	checkpoint.best_checkpoint_metric=wer \
	checkpoint.save_dir=/workspace/fairseq/scripts/whale/outputs/2022-03-30/07-39-06/checkpoints \
	dataset.valid_subset=dev-other \
	criterion._name=spk_clf \
    task.data=/workspace/LibriSpeech/manifests \
    model.w2v_path=/workspace/models/data2vec_model/audio_base_ls.pt \
	+model.viewmaker=true \
	#+model.branch_ctc_v2=true \
	#+model.overrides=uses_branch:true \
	#task.overrides=uses_branch:true \
