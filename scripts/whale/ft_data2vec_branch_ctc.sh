git pull
fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h \
	common.user_dir=examples/data2vec \
	checkpoint.best_checkpoint_metric=wer_12 \
	dataset.valid_subset=dev-other \
	criterion._name=branch_ctc \
    task.data=/home/work/workspace/LibriSpeech/manifests \
    model.w2v_path=/home/work/workspace/models/data2vec_model/audio_base_ls.pt \
	+task.uses_branch=true \
	+model.branch_ctc=true \
	#+model.overrides=uses_branch:true \
	#task.overrides=uses_branch:true \
