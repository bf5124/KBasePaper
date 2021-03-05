# KBase File Classifier

Filetype Classifier for KBase Files. 

Please follow the FileClassificationEx ipynb for a general guideline of how to run.

Steps:

1)  List your file directories in the dirs list. Data iterable will truncate all these files and turn them into a dataframe.

2) Specify your parameters for XGBoost and Tf-idf Vectorizer.

3) Split into train/val/test set using Data_splitter with the set train/test split percentage.

4) Get n-gram transformed train and validation data by running Char_vectorizer2 and set load = False to train from scratch.

5) Train XGB, SVM, MLP by setting load = False

6) Create n-gram transformed test set by running test_char_vectorizer on our test set

7) Run TestFileClassifier to get results. Threshold is a list of different cutoff thresholds you would like to test for the plotting.

--dirs '/Users/bf/Desktop/BNL2020/BioClassifierFiles/genomic.fna' '/Users/bf/Desktop/BNL2020/BioClassifierFiles/genomic.gbff' '/Users/bf/Desktop/BNL2020/BioClassifierFiles/genomic.gff' '/Users/bf/Desktop/BNL2020/BioClassifierFiles/FASTQ_truncated' '/Users/bf/Desktop/BNL2020/BioClassifierFiles/TruncFASTQ10000' '/Users/bf/Desktop/BNL2020/BioClassifierFiles/SRA_truncated' '/Users/bf/Desktop/BNL2020/BioClassifierFiles/TruncSra10000' --filetype 'fna' 'gbff' 'gff' 'fastq' 'sra' --filelength 3000 --max_depth 10 --eta 0.1 --train_rounds 40 --ngram_min 4 --ngram_max 8 --class_vocab 12 --ttsplit 0.2 --vec_load True --vec_state "tfidfcv2.pkl" --model 'xgb' --model_load True --model_state "xgb_class.pkl" --threshold 0
