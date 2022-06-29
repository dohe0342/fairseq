mode="d2v"
instance="g5"

if [ $mode == "w2v" ]
then
	for i in {0..9}
	do
		fairseq-hydra-train \
			--config-dir /opt/ml/code/fairseq/examples/wav2vec/config/finetuning \
			--config-name base_100h_aws_$instance \
			common.user_dir=examples/data2vec \
			task.data=/opt/ml/code/LibriSpeech/manifests \
			task.normalize=false \
			model.w2v_path=/opt/ml/input/data/model/wav2vec_small.pt \
			criterion._name=viewmaker \
			checkpoint.save_dir=/opt/ml/model \
			+model.viewmaker=true
		rm /opt/ml/model/crash.pt
	done
else
	for i in {0..9}
	do
		fairseq-hydra-train \
			--config-dir /opt/ml/code/fairseq/examples/wav2vec/config/finetuning \
			--config-name base_100h_aws_$instance \
			common.user_dir=examples/data2vec \
			task.data=/opt/ml/code/LibriSpeech/manifests \
			task.normalize=true \
			model.w2v_path=/opt/ml/input/data/model/audio_base_ls.pt \
			criterion._name=viewmaker \
			checkpoint.save_dir=/opt/ml/model \
			+model.viewmaker=true
		rm /opt/ml/model/crash.pt
	done
fi
