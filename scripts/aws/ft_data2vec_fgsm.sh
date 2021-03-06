#git pull

#mkdir /home/work/workspace/fairseq/scripts/whale/outputs/$1
#mkdir /opt/ml/code/fairseq/scripts/aws/output/$1
mode="d2v"
instance="p4"

if [ $mode == "w2v" ]
then
	fairseq-hydra-train \
		--config-dir /opt/ml/code/fairseq/examples/wav2vec/config/finetuning \
		--config-name base_100h_aws_$instance \
		common.user_dir=examples/data2vec \
		task.data=/opt/ml/code/LibriSpeech/manifests \
		task.normalize=false \
		model.w2v_path=/opt/ml/input/data/model/wav2vec_small.pt \
		criterion._name=ctc \
		checkpoint.save_dir=/opt/ml/model 
else
	fairseq-hydra-train \
		--config-dir /opt/ml/code/fairseq/examples/wav2vec/config/finetuning \
		--config-name base_100h_aws_$instance \
		common.user_dir=examples/data2vec \
		task.data=/opt/ml/code/LibriSpeech/manifests \
		task.normalize=true \
		model.w2v_path=/opt/ml/input/data/model/audio_base_ls.pt \
		criterion._name=ctc \
		checkpoint.save_dir=/opt/ml/model 
fi
