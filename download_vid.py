import pandas as pd
import os

from douyin_tiktok import download_func

file_df = pd.read_csv('data/data_set_label.csv')

train_df = file_df.loc[file_df['data_split'] == 'train']
test_df = file_df.loc[file_df['data_split'] == 'test']
valid_df = file_df.loc[file_df['data_split'] == 'validation']

root_folder = 'data/train/'
try:
    os.mkdir(root_folder)
except FileExistsError:
    pass

for name,link in zip(train_df['filename'],train_df['Links']):
    download_func(link,root_folder,name)

root_folder = 'data/test/'
try:
    os.mkdir(root_folder)
except FileExistsError:
    pass
for name,link in zip(test_df['filename'],test_df['Links']):
    download_func(link,root_folder,name)

root_folder = 'data/validate/'
try:
    os.mkdir(root_folder)
except FileExistsError:
    pass
for name,link in zip(valid_df['filename'],valid_df['Links']):
    download_func(link,root_folder,name)

train_df.to_csv('data/train_label.csv',index=False)
test_df.to_csv('data/test_label.csv',index=False)
valid_df.to_csv('data/validate_label.csv',index=False)
