ROOT=$(dirname $(realpath $0))
mkdir $ROOT/nltk $ROOT/glove $ROOT/bert $ROOT/data $ROOT/model

python -m nltk.downloader -d $ROOT/nltk stopwords wordnet

wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $ROOT/glove/glove.zip
unzip -j $ROOT/glove/glove.zip -d $ROOT/glove
rm $ROOT/glove/glove.zip

wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip -O $ROOT/bert/bert.zip
unzip -j $ROOT/bert/bert.zip -d $ROOT/bert
rm $ROOT/bert/bert.zip

wget https://worksheets.codalab.org/rest/bundles/0x3782ee0e5ed044d7a7e70d12553baf12/contents/blob/ -O $ROOT/data/train_dataset
wget https://worksheets.codalab.org/rest/bundles/0xb30d937a18574073903bb38b382aab03/contents/blob/ -O $ROOT/data/develop_dataset
wget https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/ -O $ROOT/data/evaluate_script