git pull
model=$1

for subset in "dev-clean" "dev-other" "test-clean" "test-other"
#for subset in "test-clean-part"
do
	echo "====================   $model // $subset   ===================="
	if [[ "$model" =~ "hubert" ]]
	then
		CUDA_VISIBLE_DEVICES=$2 python /home/work/workspace/fairseq/examples/speech_recognition/new/infer.py \
			--config-dir /home/work/workspace/fairseq/examples/hubert/config/decode \
			--config-name infer_viterbi \
			task.data=/home/work/workspace/LibriSpeech/manifests \
			task.normalize=false \
			dataset.gen_subset=$subset \
			common_eval.path=/home/work/workspace/models/hubert_model/$model 
	else
		CUDA_VISIBLE_DEVICES=$2 python /home/work/workspace/fairseq/examples/speech_recognition/new/infer.py \
			--config-dir /home/work/workspace/fairseq/examples/speech_recognition/new/conf \
			--config-name infer \
			task=audio_finetuning \
			task.data=/home/work/workspace/LibriSpeech/manifests \
			common.user_dir=examples/data2vec \
			task.labels=ltr \
			decoding.type=viterbi \
			decoding.unique_wer_file=False \
			dataset.gen_subset=$subset \
			common_eval.path=$model \
			distributed_training.distributed_world_size=1 
	fi
	echo ""
	echo ""
done
