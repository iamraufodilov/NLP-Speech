# load necessary data
import pandas as pd
import re

# load dataset
summary_data = pd.read_csv("G:/rauf/STEPBYSTEP/Data2/news_summary_text_summarization/news_summary.csv",encoding='latin-1')
raw_data = pd.read_csv("G:/rauf/STEPBYSTEP/Data2/news_summary_text_summarization/news_summary_more.csv",encoding='latin-1')
print(summary_data.head(5))
print(raw_data.head(5))

pre1 = raw_data.iloc[:, 0:2].copy()
pre2 = summary_data.iloc[:, 0:6].copy()

# To increase the intake of possible text values to build a reliable model
pre2['text'] = pre2['author'].str.cat(pre2['date'
        ].str.cat(pre2['read_more'].str.cat(pre2['text'
        ].str.cat(pre2['ctext'], sep=' '), sep=' '), sep=' '), sep=' ')

pre = pd.DataFrame()
pre['text'] = pd.concat([pre1['text'], pre2['text']], ignore_index=True)
pre['summary'] = pd.concat([pre1['headlines'], pre2['headlines']],
                           ignore_index=True)

print(pre.head(2))


# cleaning the dataset
def clean_data(collumn):
    for row in collumn:
        row = re.sub("(\\t)", " ", str(row)).lower()
        row = re.sub("(\\t)", " ", str(row)).lower()
        row = re.sub("(\\t)", " ", str(row)).lower()

        # remove the characters
        row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", " ", str(row)).lower()