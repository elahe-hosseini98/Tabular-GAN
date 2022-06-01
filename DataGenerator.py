from tabgan.sampler import OriginalGenerator, GANGenerator
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data_path = "T1w-CE-Core.xlsx"


# random input data
train = pd.DataFrame(np.random.randint(-10, 150, size=(150, 4)), columns=list("ABCD"))
target = pd.DataFrame(np.random.randint(0, 2, size=(150, 1)), columns=list("Y"))
test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))
'''
# generate data
new_train1, new_target1 = OriginalGenerator().generate_data_pipe(train, target, test, )
new_train2, new_target2 = GANGenerator().generate_data_pipe(train, target, test, )
'''
random_state = 42
data = pd.read_excel(data_path)
print(data)
X = data.drop(['label'], axis=1)
y = data.label
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
def holdout_spliting(X, y):
    X_train, X_test , y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    return X_train, X_test , y_train , y_test

X_train, X_test , y_train , y_test = holdout_spliting(X, y)


#new_train1, new_target1 = OriginalGenerator().generate_data_pipe(pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test), )
#new_train2, new_target2 = GANGenerator().generate_data_pipe(pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test),)
#print(len(new_train1))
#print(len(new_train2))  
    
           
# example with all params defined
new_train3, new_target3 = GANGenerator(gen_x_times=1.1, cat_cols=None,
           bot_filter_quantile=0.001, top_filter_quantile=0.999, is_post_process=False,
           adversarial_model_params={
               "metrics": "AUC", "max_depth": 2, "max_bin": 100, 
               "learning_rate": 0.02, "random_state": 42, "n_estimators": 500,
           }, pregeneration_frac=2, only_generated_data=False,
           gan_params = {"batch_size": 500, "patience": 25, "epochs" : 500,}).generate_data_pipe(pd.DataFrame(X_train), pd.DataFrame(y_train), 
           pd.DataFrame(X_test), deep_copy=True, only_adversarial=False, use_adversarial=True)
                                                                                                 



print(len(new_train3))



                                                                                                 