git pull

exp_name=$1

mkdir /home/work/workspace/fairseq/scripts/whale/outputs/$1
fairseq-hydra-train \
	--config-dir /workspace/fairseq/examples/wav2vec/config/finetuning \
	--config-name base_100h_summit \
	common.user_dir=examples/data2vec \
	task.data=/workspace/LibriSpeech/manifests \
	task.normalize=true \
	model.w2v_path=/workspace/models/data2vec_model/audio_base_ls.pt \
	criterion._name=ctc_fgsm \
	checkpoint.save_dir=/workspace/fairseq/scripts/whale/outputs/$1 \ 
	+model.viewmaker=true 
