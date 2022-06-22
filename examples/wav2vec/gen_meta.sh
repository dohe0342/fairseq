dir=$1
name=$2

echo "generate tsv file (from $dir to $name.tsv)"
python wav2vec_manifest.py $dir --dest $dir/manifests --ext flac --output $name

echo "generate ltr, wrd file (from $dir to $name)"
python libri_labels.py $dir/manifests/$name.tsv --output-dir $dir/manifests --output-name $name
