git pull
source ~/.bashrc
#path="/home/work/workspace/exp/viewmaker_try23_labmda_cosine_annealing_progressive_linear_growing/model"
path="/home/work/workspace/exp/viewmaker_try26_labmda_cosine_annealing_slow/model"
#path="/home/work/workspace/exp/viewmaker_try21_labmda_cosine_annealing/model"
model=$2

for subset in "dev-clean" "dev-other" "test-clean" "test-other"
do
	echo "====================   $model // $subset   ===================="
	CUDA_VISIBLE_DEVICES=$1 python /home/work/workspace/fairseq/examples/speech_recognition/new/infer.py \
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
	echo ""
	echo ""
done
