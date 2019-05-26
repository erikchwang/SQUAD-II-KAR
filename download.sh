ROOT=$(dirname $(realpath $0))
mkdir $ROOT/data $ROOT/model $ROOT/nltk
wget https://worksheets.codalab.org/rest/bundles/0x3782ee0e5ed044d7a7e70d12553baf12/contents/blob/ -O $ROOT/data/train_dataset
wget https://worksheets.codalab.org/rest/bundles/0xb30d937a18574073903bb38b382aab03/contents/blob/ -O $ROOT/data/develop_dataset
wget https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/ -O $ROOT/data/evaluate_script
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $ROOT/data/glove_archive.zip
unzip -j $ROOT/data/glove_archive.zip -d $ROOT/data
mv $ROOT/data/$(zipinfo -1 $ROOT/data/glove_archive.zip) $ROOT/data/glove_archive
rm $ROOT/data/glove_archive.zip
python -m nltk.downloader -d $ROOT/nltk stopwords wordnet