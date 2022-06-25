#git pull

exp_name=$1

#mkdir /home/work/workspace/fairseq/scripts/whale/outputs/$1
#mkdir /opt/ml/code/fairseq/scripts/aws/output/$1
fairseq-hydra-train \
	--config-dir /opt/ml/code/fairseq/examples/wav2vec/config/finetuning \
	--config-name base_100h_aws_g5 \
	common.user_dir=examples/data2vec \
	task.data=/opt/ml/code/LibriSpeech/manifests \
	task.normalize=true \
	model.w2v_path=/opt/ml/input/data/model/audio_base_ls.pt \
	criterion._name=ctc_fgsm \
	checkpoint.save_dir=/opt/ml/model/$exp_name
	#checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 
	#+model.viewmaker=true \
