git pull
source ~/.bashrc
model=$1
kenlm="true"

for subset in "dev-clean" "dev-other" "test-clean" "test-other" 
do
	echo "====================   $model // $subset   ===================="
	if [ $kenlm != "true" ]
	then
		python /workspace/fairseq/examples/speech_recognition/new/infer.py \
			--config-dir /workspace/fairseq/examples/hubert/config/decode \
			--config-name infer_viterbi \
			common_eval.path=/workspace/models/hubert_model/hubert_baseline.pt \
			common_eval.quiet=true \
			task.data=/workspace/LibriSpeech/manifests \
			task.normalize=false \
			dataset.gen_subset=$subset 
	else
		python /workspace/fairseq/examples/speech_recognition/new/infer.py \
			--config-dir /workspace/fairseq/examples/hubert/config/decode \
			--config-name infer_kenlm \
			common_eval.path=/workspace/models/hubert_model/hubert_baseline.pt \
			common_eval.quiet=true \
			task.data=/workspace/LibriSpeech/manifests \
			task.normalize=false \
			dataset.gen_subset=$subset \
			decoding.lexicon=/workspace/models/lm_model/librispeech_lexicon.lst \
			decoding.lmpath=/workspace/models/lm_model/4-gram.arpa.gz \
			decoding.lmweight=2.15 \
			decoding.wordscore=-0.52
	fi
	#model.w2v_path=/workspace/models/hubert_model/hubert_base_ls960.pt \
	echo ""
	echo ""
done
