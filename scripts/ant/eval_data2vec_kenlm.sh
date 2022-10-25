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
		decoding.type=kenlm \
		decoding.lmweight=0. decoding.wordscore=-0.52 decoding.silweight=0 \
		decoding.lexicon=/workspace/models/lm_model/librispeech_lexicon.lst \
		decoding.lmpath=/workspace/models/lm_model/4-gram.bin \
		decoding.unique_wer_file=false \
		dataset.gen_subset=$subset \
		common_eval.path=/workspace/models/wav2vec_model/wav2vec_small_100h.pt \
		common_eval.quiet=false \
	   	decoding.beam=500 \
		distributed_training.distributed_world_size=1
done
