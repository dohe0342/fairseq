git pull
fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h_whale \
	common.user_dir=examples/wav2vec \
    task.data=/home/work/workspace/LibriSpeech/manifests \
    model.w2v_path=/home/work/workspace/models/wav2vec_model/wav2vec_small.pt \
    #model.w2v_path=/home/work/workspace/fairseq/scripts/whale/multirun/2022-03-04/18-15-58/0/checkpoints/checkpoint_last.pt\
    #model.w2v_path=/home/work/workspace/models/data2vec_model/audio_base_ls.pt \
	#+model.init_transformer=true
