git pull
fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h_whale \
	common.user_dir=examples/data2vec \
	checkpoint.best_checkpoint_metric=acc \
	dataset.valid_subset=dev-other \
	criterion._name=spk_clf \
    task.data=/home/work/workspace/LibriSpeech/manifests \
    model.w2v_path=/home/work/workspace/models/data2vec_model/audio_base_ls.pt \
	+model.add_spk_info=true \
	#+model.overrides=uses_branch:true \
	#task.overrides=uses_branch:true \
