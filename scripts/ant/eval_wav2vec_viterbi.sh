model=$1
for subset in "dev-clean" "dev-other" "test-clean" "test-other"
do
	CUDA_VISIBLE_DEVICES=$3 python /workspace/fairseq/examples/speech_recognition/infer.py \
	  /workspace/LibriSpeech/manifests \
	  --task audio_pretraining \
	  --gen-subset $subset \
	  --w2l-decoder viterbi \
	  --criterion ctc \
	  --labels ltr \
	  --max-tokens 4000000 \
	  --post-process letter
done
