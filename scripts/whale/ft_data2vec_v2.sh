fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
	--config-name base_100h_whale \
	task.data=/home/work/workspace/librispeech/manifests \
	model.w2v_path=/home/work/workspace/models/wav2vec_model/wav2vec_small.pt \
	checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/ablation \
	criterion._name=viewmaker \
	+model.viewmaker=true 
