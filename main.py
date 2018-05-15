import time
from bow import OpenCVBow
import pandas as pd
# import google_api

start_time = time.time()

N = 2
bow = OpenCVBow(N)
df = pd.read_csv('./flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt', sep=" ", header=None)
df = df.drop_duplicates(0, keep='first')
logo_arr = pd.unique(df.iloc[:, 1])
print(logo_arr)
for logo in logo_arr:
    print('--------'+logo+'--------')
    bow.train(logo)
    print('----------------')

print("--- %s seconds ---" % (time.time() - start_time))
