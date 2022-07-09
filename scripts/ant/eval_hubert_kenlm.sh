git pull
source ~/.bashrc
model=$1

for subset in "dev-clean" "dev-other" "test-clean" "test-other" 
do
	echo "====================   $model // $subset   ===================="
	python /workspace/fairseq/examples/speech_recognition/new/infer.py \
		--config-dir /workspace/fairseq/examples/hubert/config/decode \
		--config-name infer_kenlm \
		task.data=/workspace/LibriSpeech/manifests \
		task.normalize=false \
		common_eval.path=/workspace/models/hubert_model/hubert_baseline.pt
		dataset.gen_subset=test \
		decoding.decoder.lexicon=/path/to/lexicon \
		decoding.decoder.lmpath=/path/to/arpa
	echo ""
	echo ""
done
