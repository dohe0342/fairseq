for model in "wav2vec_small_100h.pt" "wav2vec_small_960h.pt" "wav2vec_big_100h.pt" "wav2vec_big_960h.pt" "wav2vec_vox_100h_new.pt" "wav2vec_vox_960h_new.pt"
#for model in "wav2vec_big_960h.pt"
do
	#./eval_data2vec_viterbi.sh $model 0
	#mkdir ./None/$model
	#mv ./None/test-* ./None/$model/
	python wrong_word_sort.py $model > "$model"_to.txt
done
