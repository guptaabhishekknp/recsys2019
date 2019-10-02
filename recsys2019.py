import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import csv

df = pd.read_csv('train.csv')

df = df[['action_type','reference','impressions']]
df['action_type']=df.action_type.apply(lambda x: int((x=='clickout item') or (x=='filter selection') or (x=='change of sort order')))
df['csum']=df.action_type.cumsum()
df = df[df.reference.str.isdigit()]
df['class'] = df['csum']-df['action_type']
df = df.drop('csum',axis=1) #at this point action_type 1 is a clickout item and type 0 is anything other than clickout, filter or change sort order
df['reference'] = df.reference.apply(int) #convert the column to int for faster speed


def good_finder(x):
    if type(x)==str:
        return '|' in x
    else:
        return math.isnan(x)
    
df = df[df.impressions.apply(good_finder)] #cleaning up the data by dropping the bad ones

def make_into_list(x):
    if type(x)==str:
        return [int(i) for i in x.split('|')]
    else:
        return []

df['impressions'] = df.impressions.apply(make_into_list) 

clickouts = df[df['action_type']==1]
clickouts['legit'] = clickouts.apply(lambda x: x.reference in x.impressions, axis=1) #so you can use apply function for not just a column but an entire dataframe!!
bad_indexes = clickouts[clickouts.legit==False].index

df = df.drop(bad_indexes)
#now we have cleaned up the data and have converted as many columns to int as possible - so this dataframe is quite optimized for implementing our bayesian predictor

g = df.groupby('class')

def checker(x):
    return 1 in x.tolist()

chek = g.action_type.apply(checker)

all_interacted_items = g.reference.apply(pd.Series.tolist)[chek]
impressions_lists = ((df[df.action_type==1]).set_index('class')).impressions
res = pd.concat([all_interacted_items, impressions_lists], axis=1)

def trim(x):
    if len(x)==1:
        return x
    else:
        return x[-2:]
    
res['reference']=res.reference.apply(trim)

def indices(a,b):
    
    n = b.index(a[-1])
    if len(a)==1:
        m = 25
    else:
        if a[0] not in b:
            m = 25
        else:
            m = b.index(a[-2])
    
  
    return (m,n)

res['ind'] = res[['reference','impressions']].apply(lambda x: indices(x.reference,x.impressions),axis=1) #the axis =1 is super important!!

final = res.ind.value_counts()


A = np.zeros((26,25))

final_indices = final.index

for m in range(26):
    for n in range(25):
        if (m,n) in final_indices:
            A[m][n] = final[(m,n)]
            
plt.imshow(A)#,cmap='gray_r')
np.save('weight_matrix.npy',A)

def split_impressions(impressions):
    return impressions.split('|')

def impressions_index(reference, impressions_array):
    try:
        return impressions_array.index(reference)
    except ValueError:
        return -1

def is_interaction(row):
    return row[4].startswith("interaction item") or row[4].startswith("clickout")

def sort_impressions(histo, last_interaction, impressions):
    interaction_index = impressions_index(last_interaction, impressions)
    frequencies = histo[interaction_index + 1]
    pairs = sorted(zip(frequencies, impressions))[::-1]
    return [impression for (freq, impression) in pairs]

def is_blank_clickout(row):
    return row[4].startswith('clickout') and row[5] == ''

def is_invalidating(row):
    return (row[3] == '1' and row[4].startswith('clickout')) or (not is_interaction(row))


def build_recs(rows, histo):
    last_interaction = None

    for row in rows:
        if is_invalidating(row):
            last_interaction = None
        if is_blank_clickout(row):
            impressions = split_impressions(row[-2])
            user_id = row[0]
            session_id = row[1]
            timestamp = row[2]
            step = row[3]
            item_recommendations = sort_impressions(histo, last_interaction, impressions)
            yield [user_id, session_id, timestamp, step, ' '.join(item_recommendations)]
        elif is_interaction(row):
            last_interaction = row[5]
        
    
def write_recs(path, recs):
    with open(path, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['user_id', 'session_id', 'timestamp', 'step', 'item_recommendations'])
        for r in recs:
            writer.writerow(r)            

histo = np.vstack([A[25],A[0:25]])
with open('test.csv',encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    recs = list(build_recs(reader, histo))
    #print(recs)
    write_recs('position_recs.csv', recs)