git pull

fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
	--config-name base_100h_nospec \
	task.data=/home/work/workspace/LibriSpeech/manifests \
	task.normalize=true \
	model.w2v_path=/home/work/workspace/models/data2vec_model/audio_base_ls.pt \
	criterion._name=ctc \
	checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 
