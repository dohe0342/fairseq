git pull
source ~/.bashrc
model=$1

for subset in "dev-clean" "dev-other" "test-clean" "test-other" 
do
	echo "====================   $model // $subset   ===================="
	CUDA_VISIBLE_DEVICES=$2 python /workspace/fairseq/examples/speech_recognition/new/infer.py \
		--config-dir examples/speech_recognition/new/conf \
		--config-name infer \
		task=audio_finetuning \
		task.data=/workspace/LibriSpeech/manifests \
		common.user_dir=examples/data2vec \
		task.labels=ltr \
		decoding.type=kenlm \
		decoding.lmweight=2.13 decoding.wordscore=-0.52 decoding.silweight=0 \
		decoding.lexicon=/workspace/fairseq/lm_model/librispeech_lexicon.lst \
		decoding.lmpath=/workspace/fairseq/lm_model/4-gram.arpa.gz \
		decoding.unique_wer_file=False \
		dataset.gen_subset=$2 \
		common_eval.path=/home/work/workspace/fairseq/examples/data2vec/data2vec_model/$model \
	   	decoding.beam=1500 \
		distributed_training.distributed_world_size=1
done
