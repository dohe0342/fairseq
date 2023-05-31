git pull
subset=$1
#model=$1

for subset in $subset #"dev-clean" "dev-other" "test-clean" "test-other" 
do
	echo "====================   $model // $subset   ===================="
	python /workspace/fairseq/examples/speech_recognition/new/infer.py \
		--config-dir /workspace/fairseq/examples/speech_recognition/new/conf \
		--config-name infer \
		task=audio_finetuning \
		task.data=/workspace/LibriSpeech/manifests \
		task.labels=ltr \
		decoding.type=fairseqlm \
		decoding.lmweight=0.87 decoding.wordscore=-1.0 decoding.silweight=0 \
		decoding.lmpath=/workspace/models/lm_model/lm_librispeech_word_transformer.pt \
		decoding.lexicon=/workspace/models/lm_model/librispeech_lexicon_lower.lst \
		decoding.unique_wer_file=false \
		dataset.gen_subset=$subset \
		common_eval.path=/workspace/models/wav2vec_model/wav2vec_small_100h.pt \
		common_eval.quiet=false \
	   	decoding.beam=5 \
		distributed_training.distributed_world_size=1
done
