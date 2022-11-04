git pull
mode="w2v"
exp_name=$1

fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
	--config-name base_100h_whale \
	task.data=/home/work/workspace/LibriSpeech/manifests \
	model.w2v_path=/home/work/workspace/models/wav2vec_model/wav2vec_small.pt \
	checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 \
	criterion._name=ctc
