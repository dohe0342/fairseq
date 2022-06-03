git pull
source ~/.bashrc
model=$1

#for subset in "test" "test2" #"dev-clean" "dev-other" "test-clean" "test-other"
#for subset in "dev-clean" "dev-other" "test-clean" "test-other"
#for subset in "train-960" #"dev-clean" "dev-other" "test-clean" "test-other"
#for subset in "train-100" "train-360" "train-500" ##"dev-clean" "dev-other" "test-clean" "test-other"
#for subset in "it_was2"
for i in {0..19}
do
	first=$i
	
	$num1=1
	$num2=5
	last=$($($i + $num1) * $num2)
	echo $first $last

	#subset="test-clean_
	echo "====================   $model // $subset   ===================="
	CUDA_VISIBLE_DEVICES=$2 python /workspace/fairseq/examples/speech_recognition/new/infer.py \
		--config-dir /workspace/fairseq/examples/speech_recognition/new/conf \
		--config-name infer \
		task=audio_finetuning \
		task.data=/workspace/LibriSpeech/manifests/splited \
		common.user_dir=examples/data2vec \
		task.labels=ltr \
		decoding.type=viterbi \
		decoding.unique_wer_file=true \
		dataset.gen_subset=$subset \
		common_eval.quiet=true \
		common_eval.path=/workspace/models/wav2vec_model/$model \
		distributed_training.distributed_world_size=1 
	echo ""
done
