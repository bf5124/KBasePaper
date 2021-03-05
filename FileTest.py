import FileClassification as fc
import argparse
import os
import numpy as np
import sys

def parse_args():
    parser = argparse.ArgumentParser(description = 'File Classifier for KBase')
    parser.add_argument('--dirs', nargs = '+')
    parser.add_argument('--filetype', nargs = '+')
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
    #parser.add_argument('--train', type = bool, default = True)
    parser.add_argument('--vec_load', type = bool, default = False)
    parser.add_argument('--vec_state', type = str, default = '')
    parser.add_argument('--model', type = str, default = 'xgb')
    parser.add_argument('--model_load', type = bool, default = False)
    parser.add_argument('--model_state', type = str, default = '')
    parser.add_argument('--threshold', nargs = '+', type = float)
    
    return(parser.parse_args())
    
if __name__ == "__main__":
   
    
#--dirs '/Users/bf/Desktop/BNL2020/BioClassifierFiles/genomic.fna' '/Users/bf/Desktop/BNL2020/BioClassifierFiles/genomic.gbff' '/Users/bf/Desktop/BNL2020/BioClassifierFiles/genomic.gff' '/Users/bf/Desktop/BNL2020/BioClassifierFiles/FASTQ_truncated' '/Users/bf/Desktop/BNL2020/BioClassifierFiles/TruncFASTQ10000' '/Users/bf/Desktop/BNL2020/BioClassifierFiles/SRA_truncated'  '/Users/bf/Desktop/BNL2020/BioClassifierFiles/TruncSra10000' --filetype 'fna' 'gbff' 'gff' 'fastq' 'sra' --filelength 3000 --max_depth 10 --eta 0.1 --train_rounds 40 --ngram_min 4 --ngram_max 8 --class_vocab 12 --ttsplit 0.2 --vec_load True --vec_state "tfidfcv2.pkl" --model 'xgb' --model_load True --model_state "xgb_class.pkl" --threshold 0
    #"tfidfcv2.pkl"
    #"xgb_class.pkl"
    #"svm_class.pkl"
    #'FileClassMLP.pt'
    
    args = parse_args()
    
    print('Args Parsed')
    
    data_iterable = fc.trainloader(args.dirs, args.filetype, args.filelength)
    
    print('Data Loaded')
    
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
    filetype = args.filetype
    
    X_train, X_val, X_test, y_train, y_val, y_test = fc.Data_Splitter(Dataset, ttsplit)

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
        
        if args.model == 'xgb':
            model = fc.TrainXGBClassifier(param, num_round, dat_train, dat_val, y_train, y_val, load = True, state = args.model_state)
            
        if args.model == 'svm':
            model = fc.TrainSVMClassifier(dat_train, y_train, load = True, state= args.model_state)
            
        if args.model == 'mlp':
            model = fc.TrainMLPClassifier(FileNet, dat_train, y_train, dat_val, y_val, epochs = args.train_rounds, load = True, state = args.model_state )
    
    
    dat_test = fc.test_char_vectorizer(X_test, state = args.vec_state)
    
    preds = fc.TestFileClassifier(model, dat_test, filetype, y_test, output = False, threshold = args.threshold, threshold_plots = False, classifier = args.model)
    
    print(preds)
    
    return(preds)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
