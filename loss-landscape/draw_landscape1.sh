git pull
mpirun -n 8 python plot_surface.py \
	--mpi --cuda --model wav2vec --dataset LibriSpeech \
	--x=-0.3:0.3:5 --y=-0.3:0.3:5 \
	--model_file /home/work/workspace/models/wav2vec_model/wav2vec_small_960h.pt \
	--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn \
	--plot --loss_name ctc --subset $1 --root_rank $2
