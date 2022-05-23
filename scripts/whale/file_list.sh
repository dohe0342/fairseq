#files=(/home/work/workspace/exp/viewmaker_try24_lambda_cosine_annealing_progressive_linear_growing/model/*)
#echo $files
declare -a arrPics
for file in *.jpg
do
    arrPics=("${Pics[@]}" "$file")
done
