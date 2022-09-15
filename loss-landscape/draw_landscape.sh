git pull
mpirun -n 8 python plot_surface.py --mpi --cuda --model resnet56 --dataset LibriSpeech --x=-1:1:10 --y=-1:1:10 --model_file /home/work/workspace/models/wav2vec_model/wav2vec_small_100h.pt --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot --loss_name ctc --subset dev-clean 
