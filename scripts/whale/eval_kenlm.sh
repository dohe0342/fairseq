for subset in "dev-clean" "dev-other" "test-clean" "test-other"
do
	python /home/work/workspace/fairseq/examples/speech_recognition/new/infer.py \
		--config-dir /home/work/workspace/fairseq/examples/speech_recognition/new/conf \
		--config-name infer \
		task=audio_finetuning \
		task.data=/home/work/workspace/LibriSpeech/manifests \
		common.user_dir=/home/work/workspace/fairseq/examples/data2vec \
		task.labels=ltr \
		decoding.type=kenlm \
		decoding.lmweight=2.0 \ 
		decoding.wordscore=-0.3 \
		decoding.silweight=0.0 \
		decoding.lexicon=/home/work/workspace/models/lm/librispeech-lexicon.txt \
		decoding.lmpath=/home/work/workspace/models/lm/4-gram.arpa.gz \
		decoding.unique_wer_file=False \
		dataset.gen_subset=$subset \
		common_eval.path=/home/work/workspace/models/wav2vec_model/wav2vec_small_100h.pt \
		decoding.beam=500 \
		distributed_training.distributed_world_size=1
done
