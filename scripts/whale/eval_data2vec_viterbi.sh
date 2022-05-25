git pull
source ~/.bashrc
path=/home/work/workspace/exp/viewmaker_try23_labmda_cosine_annealing_progressive_linear_growing/model
model=$1

for subset in "dev-clean" "dev-other" "test-clean" "test-other"
do
	for i in {2..10}
	do
		echo "====================   $model // $subset // $i idx pruned   ===================="
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
			common_eval.path=/home/work/workspace/models/data2vec_model/$model \
			distributed_training.distributed_world_size=1 \
			+model.ch_prune_idx=2
		echo ""
		echo ""
	done
done
