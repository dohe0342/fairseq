for subset in "dev-clean" "dev-other" "test-clean" "test-other"
do
	python /home/work/workspace/fairseq/examples/speech_recognition/new/infer.py \
		--config-dir examples/speech_recognition/new/conf \
		--config-name infer \
		task=audio_finetuning \
		task.data=/home/work/workspace/LibriSpeech/manifests \
		common.user_dir=/home/work/workspace/fairseq/examples/data2vec \
	task.labels=ltr decoding.type=kenlm \
	decoding.lmweight=${lmweight} decoding.wordscore=${wordscore} decoding.silweight=${silscore} \
	decoding.lexicon=/path/to/lexicon \
	decoding.lmpath=/path/to/lm decoding.unique_wer_file=True \
	dataset.gen_subset=dev_clean,dev_other,test_clean,test_other \
	common_eval.path=/path/to/checkpoint.pt decoding.beam=1500 distributed_training.distributed_world_size=${num_gpus}
done
