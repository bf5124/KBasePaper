# FileClassificationReal

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