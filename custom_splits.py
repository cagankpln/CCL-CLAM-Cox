import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--data_slide_dir', type=str, default=None, help='Path to slide directory')
parser.add_argument('--splits_csv', type=str, default=None, help='Path to splits csv')
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
args = parser.parse_args()

slide_names = os.listdir(args.data_slide_dir)

slide_names_cohort1 = [] #train and validation set
slide_names_cohort2 = [] #test set
for slide in slide_names:
    if "cohort1" in slide:
        slide_names_cohort1.append(slide)
    else:
        slide_names_cohort2.append(slide)

train_files, val_files = train_test_split(slide_names_cohort1, test_size=args.val_frac, random_state=42)

df = pd.DataFrame()

df['train'] = train_files
df['val'] = val_files + [None] * (len(df['train']) - len(val_files))
df["test"] = slide_names_cohort2 + [None] * (len(df['train']) - len(slide_names_cohort2))

df.to_csv(args.splits_csv)
