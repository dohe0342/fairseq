git pull

exp_name=$1

mkdir /workspace/fairseq/scripts/ant/outputs/$1
fairseq-hydra-train \
	--config-dir /workspace/fairseq/examples/wav2vec/config/finetuning \
	--config-name base_100h_ant \
	common.user_dir=examples/data2vec \
	task.data=/workspace/LibriSpeech/manifests \
	task.normalize=true \
	model.w2v_path=/workspace/models/data2vec_model/audio_base_ls.pt \
	criterion._name=ctc \
	checkpoint.save_dir=/workspace/fairseq/scripts/ant/outputs/$1
