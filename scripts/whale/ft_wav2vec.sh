git pull
fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h_whale \
	common.user_dir=examples/wav2vec \
    task.data=/home/work/workspace/LibriSpeech/manifests \
    model.w2v_path=/home/work/workspace/models/wav2vec_model/wav2vec_small.pt 
