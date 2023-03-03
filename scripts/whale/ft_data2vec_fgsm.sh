git pull

exp_name=$1

mkdir /home/work/workspace/fairseq/scripts/whale/outputs/$1
fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
	--config-name base_100h_whale \
	common.user_dir=examples/data2vec \
	task.data=/home/work/workspace/LibriSpeech/manifests \
	task.normalize=false \
	model.w2v_path=/home/work/workspace/models/wav2vec_model/wav2vec_small.pt \
	criterion._name=ctc_fgsm \
	checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 
	#+model.viewmaker=true \
	#model.w2v_path=/home/work/workspace/models/data2vec_model/audio_base_ls.pt \
