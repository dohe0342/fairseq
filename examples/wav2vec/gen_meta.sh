dir=$1
name=$2

echo "generate tsv file (from $dir to $name.tsv)"
python wav2vec_manifest.py $dir --dest $dir --ext flac --output $name

echo "generate ltr, wrd file (from $dir to $name)"
python libri_labels.py $dir/$name.tsv --output-dir $dir --output-name $name

