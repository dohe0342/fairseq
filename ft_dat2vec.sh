fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h \
	common.user_dir=examples/data2vec \
    task.data=/home/work/workspace/LibriSpeech/manifests \
    model.w2v_path=/home/work/workspace/fairseq/examples/data2vec/data2vec_model/audio_base_ls.pt \
	model.freeze_finetune_updates=80000
    
