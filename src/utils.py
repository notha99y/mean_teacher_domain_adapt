import os
import pandas as pd
import seaborn as sns


def get_distribution(data_path, label, color):
    walker = os.walk(data_path)
    next(walker)  # skip the first row
    class_freq = dict()
    for r, d, f in walker:
        class_freq[r.split('/')[-1]] = len(f)
    class_freq_df = pd.DataFrame.from_dict(
        class_freq, orient='index', columns=['count'])
    class_freq_df.reset_index(inplace=True)
    class_freq_df.columns = [label, 'count']
    class_freq_df.sort_values('count', axis=0, ascending=False, inplace=True)

    sns.catplot(x='count', y=label, kind='bar',
                data=class_freq_df, color=color)
