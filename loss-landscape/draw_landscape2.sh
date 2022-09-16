git pull
mpirun -n 4 python plot_surface.py \
    --mpi --cuda --model ctc --dataset LibriSpeech \
    --x=-0.2:0.2:5 --y=-0.2:0.2:5 \
    --model_file /home/work/workspace/fairseq/scripts/whale/outputs/w2v_b_960h_mba3f/wav2vec_small_960h_mba.pt \
    --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn \
    --plot --loss_name ctc --subset $1 --root_rank $2 
