from ctgan import CTGANSynthesizer
from ctgan import load_demo
import pandas as pd
import os

dir_list = os.listdir()
for file_name in dir_list:
    if file_name.endswith(".xlsx"):
        data = pd.read_excel(file_name)
        if 'Tag' in data: 
            drop_elements = ['Tag']
            data = data.drop(drop_elements, axis = 1)
        
        ctgan = CTGANSynthesizer(epochs=20)
        ctgan.fit(data, ['label'])
        
        # Synthetic copy
        new_samples = ctgan.sample(200)
        
        full_dataset = [data, new_samples]
        result = pd.concat(full_dataset)
        result.to_excel("new_" + file_name)



