git pull
mpirun -n 8 python plot_surface.py --mpi --cuda --model resnet56 --dataset LibriSpeech --x=-1:1:10 --y=-1:1:10 --model_file /home/work/workspace/fairseq/scripts/whale/outputs/w2v_b_960h_mba3f/checkpoint_best.pt --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot --loss_name ctc --subset dev-clean 
