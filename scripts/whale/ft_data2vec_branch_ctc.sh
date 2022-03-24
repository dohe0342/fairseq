git pull
fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h_whale \
	common.user_dir=/home/work/workspace/fairseq/examples/data2vec \
	checkpoint.best_checkpoint_metric=wer \
	checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/2022-03-21/d2v_mid1_spkclf_reverse_1.0
	dataset.valid_subset=dev-other \
	criterion._name=spk_clf \
    task.data=/home/work/workspace/LibriSpeech/manifests \
    model.w2v_path=/home/work/workspace/models/data2vec_model/audio_base_ls.pt \
	+model.del_spk_info=true \
	#+model.branch_ctc_v2=true \
	#+model.overrides=uses_branch:true \
	#task.overrides=uses_branch:true \
