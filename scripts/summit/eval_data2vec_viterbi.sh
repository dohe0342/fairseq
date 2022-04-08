git pull
source ~/.bashrc
model=$1

for subset in "train-100" #"dev-clean" "dev-other" "test-clean" "test-other"
do
	echo "====================   $model // $subset   ===================="
	CUDA_VISIBLE_DEVICES=$2 python /home/work/workspace/fairseq/examples/speech_recognition/new/infer.py \
		--config-dir /home/work/workspace/fairseq/examples/speech_recognition/new/conf \
		--config-name infer \
		task=audio_finetuning \
		task.data=/workspace/LibriSpeech/manifests \
		common.user_dir=examples/data2vec \
		task.labels=ltr \
		decoding.type=viterbi \
		decoding.unique_wer_file=False \
		dataset.gen_subset=$subset \
		common_eval.path=/workspace/models/data2vec_model/$model \
		distributed_training.distributed_world_size=1
	echo ""
	echo ""
done
