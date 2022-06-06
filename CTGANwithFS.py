from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, ShuffleSplit, StratifiedShuffleSplit, cross_val_score, cross_val_predict,
                                     KFold, StratifiedKFold, RandomizedSearchCV, GridSearchCV, learning_curve)
from sklearn.metrics import (mean_squared_error, accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score,
                             confusion_matrix, precision_score, recall_score, f1_score)

import warnings
from sklearn.utils import resample
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.linear_model import (LinearRegression, Lasso, ElasticNet, LogisticRegression,
                                  PassiveAggressiveClassifier, RidgeClassifier)
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier,
                              RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from ctgan import CTGANSynthesizer
from ctgan import load_demo
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow.keras as K

warnings.filterwarnings("ignore")
plt.rcdefaults()
random_state = 42

class FS_and_CLF:
    def __init__(self, excel_file, ml_algo_list, fs_algo_list):
        self.excel_file = excel_file
        self.data = pd.read_excel(excel_file)
        if 'Tag' in self.data: 
            drop_elements = ['Tag']
            self.data = self.data.drop(drop_elements, axis = 1)
        self.X = self.data.drop(['label'], axis=1)
        self.y = self.data.label
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)
        self.ml_algo_list = ml_algo_list
        self.fs_algo_list = fs_algo_list
        self.result_df = pd.DataFrame(columns=['ML','FS', 'AUC', 'ACC', 'SEN', 'SPE', 'PPV', 'NPV'])
        
    def holdout_spliting(self):
        X_train, X_test , y_train , y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=random_state)
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        return X_train, X_test , y_train , y_test
    
    def calc_metrics(self, y_true, y_pred):
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)
        TP = confusion[1,1]
        TN = confusion[0,0]
        FP = confusion[0,1]
        FN = confusion[1,0]
        SEN = TP / (TP + FN)
        SPE = TN / (TN + FP)
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        return auc, acc, SEN, SPE, PPV, NPV
    
    def k_best(self, X_train, X_test , y_train , y_test):
        print('K-Best FS...')
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn import set_config
        set_config(display="diagram")
        nof_list=np.arange(1, 20)            
        high_score = 0
        nof = 0           
        score_list =[]
        selected_col_lists = []
        
        for ml_algo in self.ml_algo_list:
            for n in nof_list:
                fs = SelectKBest(f_classif, k=n)
                relief = Pipeline([('fs', fs), ('m', ml_algo())])
                relief.fit(X_train, y_train)
                score = relief.score(X_test, y_test)
                score_list.append(score)
                if(score > high_score):
                    high_score = score
                    nof = n
            selector = SelectKBest(f_classif, k=nof)
            selector.fit(X_train, y_train)
            cols = selector.get_support(indices=True)
            selected_col_lists.append(cols)

        longest_selected_col = max(selected_col_lists, key=len)
        X_train = X_train[:][:, longest_selected_col]
        X_test = X_test[:][:, longest_selected_col]
        ds_train = np.column_stack([X_train, y_train])
        df_columns_name = [chr(ord('A') + i) for i in range(len(longest_selected_col))] + ['label']
        df_train = pd.DataFrame(ds_train,columns=df_columns_name)
        ctgan = CTGANSynthesizer(epochs=20)
        ctgan.fit(df_train, ['label'])
        print('Generating Augmented Data...')
        new_samples = ctgan.sample(200)
        full_train_dataset = [df_train, new_samples]
        result = pd.concat(full_train_dataset)
        result = result.to_numpy()
        X_train = result[:, :-1]
        y_train = result[:, -1]
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        callback = K.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model = Sequential()
        model.add(Dense(20, input_dim=len(longest_selected_col), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(15, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val,y_val), verbose=0, callbacks=[callback])
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        auc, acc, SEN, SPE, PPV, NPV = self.calc_metrics(y_test, y_pred)
        print(acc)
        
        for ml_algo in self.ml_algo_list:      
            clf = ml_algo()
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            auc, acc, SEN, SPE, PPV, NPV = self.calc_metrics(y_test, y_pred)
            result_data = {'ML':ml_algo.__name__, 'FS':'k_best',
                           'AUC':auc, 'ACC':acc, 'SEN':SEN, 'SPE':SPE, 'PPV':PPV, 'NPV':NPV}
            
            self.result_df = self.result_df.append(result_data, ignore_index=True)
     
    def RFE(self, X_train, X_test , y_train , y_test):
        print("RFE FS...")
        from sklearn.feature_selection import RFE
        nof_list=np.arange(1, 20)            
        high_score = 0
        nof = 0           
        score_list =[]
        selected_col_lists = []
        for ml_algo in self.ml_algo_list:
            print(ml_algo.__name__)
            if not(ml_algo.__name__=='BaggingClassifier' or ml_algo.__name__=='BernoulliNB'
                   or ml_algo.__name__=='GaussianNB' or ml_algo.__name__=='HistGradientBoostingClassifier'
                   or ml_algo.__name__=='KNeighborsClassifier' or ml_algo.__name__=='MLPClassifier'
                  or ml_algo.__name__=='NearestCentroid' or
                   ml_algo.__name__=='QuadraticDiscriminantAnalysis' or ml_algo.__name__=='SVC'):
                for n in nof_list:
                    fs = RFE(ml_algo(), n_features_to_select = n)
                    fit = fs.fit(X_train, y_train)
                    score = fit.score(X_test, y_test)
                    score_list.append(score)
                    if(score > high_score):
                        high_score = score
                        nof = n
                fs = RFE(ml_algo(), n_features_to_select = nof)
                fit = fs.fit(X_train, y_train)
                cols = []
                for i in range(len(fit.support_)):
                    if fit.support_[i]:
                        cols.append(i)
                selected_col_lists.append(cols)        
         
        longest_selected_col = max(selected_col_lists, key=len)
        X_train = X_train[:][:, longest_selected_col]
        X_test = X_test[:][:, longest_selected_col]
        ds_train = np.column_stack([X_train, y_train])
        df_columns_name = [chr(ord('A') + i) for i in range(len(longest_selected_col))] + ['label']
        df_train = pd.DataFrame(ds_train,columns=df_columns_name)
        ctgan = CTGANSynthesizer(epochs=20)
        ctgan.fit(df_train, ['label'])
        print('Generating Augmented Data...')
        new_samples = ctgan.sample(200)
        full_train_dataset = [df_train, new_samples]
        result = pd.concat(full_train_dataset)
        result = result.to_numpy()
        X_train = result[:, :-1]
        y_train = result[:, -1]
        
        for ml_algo in self.ml_algo_list:
            clf = ml_algo()
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            auc, acc, SEN, SPE, PPV, NPV = self.calc_metrics(y_test, y_pred)
            result_data = {'ML':ml_algo.__name__, 'FS':'RFE',
                           'AUC':auc, 'ACC':acc, 'SEN':SEN, 'SPE':SPE, 'PPV':PPV, 'NPV':NPV}
            
            self.result_df = self.result_df.append(result_data, ignore_index=True)
      
    def mRmR(self, X_train, X_test , y_train , y_test):
        print("MRMR fs...")
        from mrmr import mrmr_classif
        nof_list=np.arange(2, 20)            
        high_score = 0      
        selected_col_lists = []
        for ml_algo in self.ml_algo_list:
            for n in nof_list:
                cols = mrmr_classif(pd.DataFrame(X_train), pd.Series(y_train), K=n)
                ml = ml_algo()
                ml.fit(X_train[:, cols], y_train)
                y_pred = ml.predict(X_test[:, cols])
                score = accuracy_score(y_test, y_pred)
                if(score > high_score):
                    high_score = score
                    best_faetures = cols
            selected_col_lists.append(best_faetures)
            
        longest_selected_col = max(selected_col_lists, key=len)
        X_train = X_train[:][:, longest_selected_col]
        X_test = X_test[:][:, longest_selected_col]
        ds_train = np.column_stack([X_train, y_train])
        df_columns_name = [chr(ord('A') + i) for i in range(len(longest_selected_col))] + ['label']
        df_train = pd.DataFrame(ds_train,columns=df_columns_name)
        ctgan = CTGANSynthesizer(epochs=20)
        ctgan.fit(df_train, ['label'])
        print('Generating Augmented Data...')
        new_samples = ctgan.sample(200)
        full_train_dataset = [df_train, new_samples]
        result = pd.concat(full_train_dataset)
        result = result.to_numpy()
        X_train = result[:, :-1]
        y_train = result[:, -1]
        
        for ml_algo in self.ml_algo_list:
            clf = ml_algo()
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            auc, acc, SEN, SPE, PPV, NPV = self.calc_metrics(y_test, y_pred)
            result_data = {'ML':ml_algo.__name__, 'FS':'RFE',
                           'AUC':auc, 'ACC':acc, 'SEN':SEN, 'SPE':SPE, 'PPV':PPV, 'NPV':NPV}
            
            self.result_df = self.result_df.append(result_data, ignore_index=True)
            
    def main(self):
        X_train, X_test , y_train , y_test = self.holdout_spliting()
        for fs_algo in self.fs_algo_list:
                if fs_algo == 'boruto':
                    self.boruta(X_train, X_test , y_train , y_test)
                elif fs_algo == 'k_best':
                    self.k_best(X_train, X_test , y_train , y_test)   
                elif fs_algo == 'RReliefF':
                    self.RReliefF(X_train, X_test , y_train , y_test)   
                elif fs_algo == 'RFE':
                    self.RFE(X_train, X_test , y_train , y_test)   
                elif fs_algo == 'mRmR':
                    self.mRmR(X_train, X_test , y_train , y_test)
        self.result_df.to_excel(str(Path().absolute())+'\\result-'+self.excel_file, index=False)
              
if __name__ == "__main__":        
    file_path = r'T2w-FLAIR-Core.xlsx'    
    fs_algo_list = ['k_best', 'RFE', 'mRmR']
    ml_algo_list = [AdaBoostClassifier,
                    BaggingClassifier,
                    BernoulliNB,
                    DecisionTreeClassifier,
                    ExtraTreesClassifier,
                    GaussianNB,
                    GradientBoostingClassifier,
                    HistGradientBoostingClassifier,
                    KNeighborsClassifier,
                    LinearDiscriminantAnalysis,
                    LogisticRegression,
                    MLPClassifier,
                    NearestCentroid,
                    PassiveAggressiveClassifier,
                    QuadraticDiscriminantAnalysis,
                    RandomForestClassifier,
                    RidgeClassifier,
                    SVC
                   ]   
    clf = FS_and_CLF(file_path, ml_algo_list, fs_algo_list)
    clf.main()

        
        

