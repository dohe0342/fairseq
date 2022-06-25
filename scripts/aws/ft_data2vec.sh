for i in {0..9} ; do
	fairseq-hydra-train \
		--config-dir /opt/ml/code/fairseq/examples/wav2vec/config/finetuning \
		--config-name base_100h_aws_p4 \
		common.user_dir=examples/data2vec \
		task.data=/opt/ml/code/LibriSpeech/manifests \
		task.normalize=true \
		model.w2v_path=/opt/ml/input/data/data/audio_base_ls.pt \
		criterion._name=viewmaker \
		checkpoint.save_dir=/opt/ml/model \
		dataset.max_tokens=3200000 \
		optimization.update_freq=[2] \
		+model.viewmaker=true
done
