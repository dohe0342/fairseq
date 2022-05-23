#files=(/home/work/workspace/exp/viewmaker_try24_lambda_cosine_annealing_progressive_linear_growing/model/*)
#echo $files
declare -a arrPics
for file in /home/work/workspace/exp/viewmaker_try24_lambda_cosine_annealing_progressive_lienar_growing/model/*.pt
do
    arrPics=("${Pics[@]}" "$file")
done

echo $arrPics
