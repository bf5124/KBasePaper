import FileClassification as fc
import argparse
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'File Classifier for KBase')
    parser.add_argument('--dirs')
    parser.add_argument('--filetype')
    parser.add_argument('--filelength', type = int, default = 3000)
    parser.add_argument('--max_depth', type = int, default = 10)
    parser.add_argument('--eta', type = float, default = 0.1)
    #parser.add_argument('--rate_drop', type = float, default = 0.5)
    #parser.add_argument('--skip_drop', type = float, default = 0.4)
    parser.add_argument('--train_rounds', type = int, default = 40)
    parser.add_argument('--ngram_min', type = int, default = 4)
    parser.add_argument('--ngram_max', type = int, default = 8)
    parser.add_argument('--class_vocab', type = int, default = 12)
    parser.add_argument('--ttsplit', type = float, default = 0.2)
    parser.add_argument('--vec_load', type = bool, default = False)
    parser.add_argument('--vec_state', type = str, default = '')
    parser.add_argument('--model', type = str, default = 'xgb')
    parser.add_argument('--model_load', type = bool, default = False)
    parser.add_argument('--model_state', type = str, default = '')
    parser.add_argument('--threshold', default = [0])
    args = parser.parse_args()
    
    #"tfidfcv2.pkl"
    #"xgb_class.pkl"
    #"svm_class.pkl"
    #'FileClassMLP.pt'
    
    data_iterable = fc.trainloader(args.dirs, args.filetype, args.filelength)
    
    
    param = { # XGB Parameters
    'max_depth': args.max_depth,  # the maximum depth of each tree
    'eta': args.eta,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': len(args.filetype),  # the number of classes that exist in this datset
    'booster' : 'dart', # Dropout added
    'rate_drop' : 0.5, #Dropout Rate
    'skip_drop' : 0.4, # Probability of skipping Dropout
    } 
    num_round = args.train_rounds # Number of rounds we train our XGB Classifier
    ngram_range = (args.ngram_min, args.ngram_max) # Character length we search by for Tfidf
    max_features = args.class_vocab # Vocab Words Per File Class
    Dataset = data_iterable
    ttsplit = args.ttsplit  # Train/Test Split Percentage
    
    X_train, X_val, X_test, y_train, y_val, y_test = Data_Splitter(Dataset, ttsplit)
    
    if args.vec_load == False:
            dat_train, dat_val = fc.Char_vectorizer2(X_train, y_train, X_val, y_val, args.filetype, ngram_range, max_features, load = False, state = args.vec_state)
    
    if args.vec_load == True:
            dat_train, dat_val = fc.Char_vectorizer2(X_train, y_train, X_val, y_val, args.filetype, ngram_range, max_features, load = True, state = args.vec_state)
    
    if args.model_load == False:
    
        if args.model == 'xgb':
            model = fc.TrainXGBClassifier(param, num_round, dat_train, dat_val, y_train, y_val, load = False, state = args.model_state)
            
        if args.model == 'svm':
            model = fc.TrainSVMClassifier(dat_train, y_train, load = False, state= args.model_state)
            
        if args.model == 'mlp':
            model = fc.TrainMLPClassifier(FileNet, dat_train, y_train, dat_val, y_val, epochs = args.train_rounds, load = False, state = args.model_state )
    
    
    if args.model_load == True:
        dat_train, dat_val = fc.Char_vectorizer2(X_train, y_train, X_val, y_val, args.filetype, ngram_range, max_features, load = True)
        if args.model == 'xgb':
            model = fc.TrainXGBClassifier(param, num_round, dat_train, dat_val, y_train, y_val, load = True, state = args.model_state)
            
        if args.model == 'svm':
            model = fc.TrainSVMClassifier(dat_train, y_train, load = True, state= args.model_state)
            
        if args.model == 'mlp':
            model = fc.TrainMLPClassifier(FileNet, dat_train, y_train, dat_val, y_val, epochs = args.train_rounds, load = True, state = args.model_state )
    
    
    dat_test = test_char_vectorizer(X_test, state = args.vec_state)
    
    TestFileClassifier(model, dat_test, filetype, y_test, output = False, threshold = args.threshold, threshold_plots = False, classifier = args.model)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
