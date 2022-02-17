model=$1
python ../examples/speech_recognition/new/infer.py --config-dir ../examples/speech_recognition/new/conf \
	--config-name infer task=audio_finetuning task.data=/home/work/workspace/LibriSpeech/manifests common.user_dir=examples/data2vec \
	task.labels=ltr decoding.type=viterbi \
	decoding.unique_wer_file=False \
	dataset.gen_subset=dev-clean \
	common_eval.path=/home/work/workspace/models/data2vec_model/$model decoding.beam=5 distributed_training.distributed_world_size=1
