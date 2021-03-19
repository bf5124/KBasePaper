# KBase File Classifier

Filetype Classifier for KBase Files.

Data files used for the model here are available at https://drive.google.com/file/d/17VhkOTuFDc52smyVyPKxIcLo5Q0UuR-m/view?usp=sharing in truncated form.

Please follow the FileClassificationEx ipynb for a general guideline of how to run.

Steps:

List your file directories in the dirs list. Data iterable will truncate all these files and turn them into a dataframe.

Specify your parameters for XGBoost and Tf-idf Vectorizer.

Split into train/val/test set using Data_splitter with the set train/test split percentage.

Get n-gram transformed train and validation data by running Char_vectorizer2 and set load = False to train from scratch.

Train XGB, SVM, MLP by setting load = False

Create n-gram transformed test set by running test_char_vectorizer on our test set

Run TestFileClassifier to get results. Threshold is a list of different cutoff thresholds you would like to test for the plotting.

DATADIR="path-to-data-directories"

python FileTest.py --dirs '$DATADIR/genomic.fna' '$DATADIR/genomic.gbff' '$DATADIR/genomic.gff' '$DATADIR/FASTQ_truncated'
'$DATADIR/TruncFASTQ10000' '$DATADIR/SRA_truncated' '$DATADIR/TruncSra10000'
--filetype 'fna' 'gbff' 'gff' 'fastq' 'sra' --filelength 3000
--max_depth 10 --eta 0.1 --train_rounds 40 --ngram_min 4 --ngram_max 8 --class_vocab 12 --ttsplit 0.2 --vec_load True --vec_state "tfidfcv2.pkl" --model 'xgb' --model_load True --model_state "xgb_class.pkl" --threshold 0
