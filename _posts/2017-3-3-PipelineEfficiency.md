---
layout: post
title: I've set up a blog!
date: 2017-12-01
published: true
categories: Blog
---

In this project I seek to make GridSearching a Pipelone more efficient for the way I like to format my pipelines.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion

import itertools

import concurrent.futures
from itertools import repeat

from sklearn.base import is_classifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection._split import check_cv

from sklearn.externals.joblib import Parallel, delayed

from sklearn.base import clone

from sklearn.model_selection._search import BaseSearchCV

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

## The Elegance and Inefficiencies of GridSeaching a Pipeline

### The Set Up/Problem

The idea for this comes from my growing habit of using pipelines to organize my data modeling process, as well as using it as a why to try out many different parameters and methodologies at once. Now this by it's nature is a time intensive process as it is creating many thousands (based on how many parameters you are trying) of models, but I also realized that due to the way in which I format and search my pipelines, I was aggravating the problem.

To understand where the issue comes into play, I will illlustrate the framework I use for pipelines below.

* **Pipeline**
    * **Feature Union** - This Feature Union holds pipelines for cleaning/transforming all the data I want to use for the model.
    * **Middle Steps** - Whether this is scaling all the data, or a form of feature selection.
    * **Modeling** - This is the final step of the pipeline where the data is given to the model to be used.
    
    **Insert Drawing Here**
    
The issue here is in the Feauture Union. Since I like to use the pipeline to clean my data, I believe there is time lost as for every variation of a model that is tried the entire pipeline is running which means that the data, which has already been fit, is getting fit again everytime anyway. While this has most likely had trivial impact for my uses so far, using simple cleaning methods like mapping values, as I've moved into more intensive processes like Natural Languange Processing I think this becomes a much more significant time issue.

Now a simple solution would be to remove the Feature Union from the pipeline, fit it seperately, and then use the remaining pipeline for the gridsearch, however, that loses the ability to also easily GridSearch pieces of the data cleaning process.

So my goal is to confirm that the problem I think should exits truly has impact, and then implement my own estimator class as a wrapper for GridSearch to better optimize for my specific use case and pipeline format.

#### Sample Data

I will be using a consistent set of data to "Economic News" data to run these tests. I will not do a Train-Test split because I really don't care about the actual results of the model here, just the tiem it takes to fit.


```python
df = pd.read_csv('sample_data/economic_news.csv', usecols=[7, 11, 14])
df.text = df.text.apply(lambda x: x.replace('</br>', ''))
df.relevance = df.relevance.apply(lambda x: 1 if x == 'yes' else 0)
print(df.shape)
df.head()
```

    (8000, 3)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>relevance</th>
      <th>headline</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Yields on CDs Fell in the Latest Week</td>
      <td>NEW YORK -- Yields on most certificates of dep...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>The Morning Brief: White House Seeks to Limit ...</td>
      <td>The Wall Street Journal OnlineThe Morning Brie...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Banking Bill Negotiators Set Compromise --- Pl...</td>
      <td>WASHINGTON -- In an effort to achieve banking ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Manager's Journal: Sniffing Out Drug Abusers I...</td>
      <td>The statistics on the enormous costs of employ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Currency Trading: Dollar Remains in Tight Rang...</td>
      <td>NEW YORK -- Indecision marked the dollar's ton...</td>
    </tr>
  </tbody>
</table>
</div>



### Proving the Problem

In this section I will look at the amount of time it takes for a few types of data cleaning/transofrmation of different complexities, and how they scale on their own and within GridSearches in order to see how much time constantly refitting data is costing.


```python
#set up some stuff

#of times to GridSearch something
reps = [1,5,10,25,50,75,100,150,200,250,500,750,1000,2500,5000]

tmp = []
df['nums'] = df.index
rs = 779
np.random.seed(rs)
df.nums = df.nums.apply(lambda x: 100*x* np.random.rand())
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>relevance</th>
      <th>headline</th>
      <th>text</th>
      <th>nums</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Yields on CDs Fell in the Latest Week</td>
      <td>NEW YORK -- Yields on most certificates of dep...</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>The Morning Brief: White House Seeks to Limit ...</td>
      <td>The Wall Street Journal OnlineThe Morning Brie...</td>
      <td>72.259737</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Banking Bill Negotiators Set Compromise --- Pl...</td>
      <td>WASHINGTON -- In an effort to achieve banking ...</td>
      <td>184.913663</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Manager's Journal: Sniffing Out Drug Abusers I...</td>
      <td>The statistics on the enormous costs of employ...</td>
      <td>94.107106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Currency Trading: Dollar Remains in Tight Rang...</td>
      <td>NEW YORK -- Indecision marked the dollar's ton...</td>
      <td>336.160412</td>
    </tr>
  </tbody>
</table>
</div>




```python
eff_orig = pd.DataFrame(reps, columns=['reps'])
eff_orig['fits'] = eff_orig.reps.apply(lambda x: x*2)
eff_orig
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reps</th>
      <th>fits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>100</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75</td>
      <td>150</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>200</td>
    </tr>
    <tr>
      <th>7</th>
      <td>150</td>
      <td>300</td>
    </tr>
    <tr>
      <th>8</th>
      <td>200</td>
      <td>400</td>
    </tr>
    <tr>
      <th>9</th>
      <td>250</td>
      <td>500</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>750</td>
      <td>1500</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1000</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2500</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5000</td>
      <td>10000</td>
    </tr>
  </tbody>
</table>
</div>




```python
eff_cached = eff_orig.copy()
```


```python
eff_ss = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('pre-process', StandardScaler())
    ])
    pipe = Pipeline([
        ('fu', fu),
        ('model', LogisticRegression())
    ])
    params = {
        'model__random_state':range(0,x)
    }
    gs = GridSearchCV(pipe, params, verbose=1, n_jobs=-1, cv=2)
    
    #start timer
    start = datetime.now()
    
    gs.fit(df.nums.values.reshape(-1,1), df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_ss.append(diff.total_seconds())
    
eff_orig['ss_orig'] = eff_ss
```


```python
eff_pf = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('pre-process', PolynomialFeatures())
    ])
    pipe = Pipeline([
        ('fu', fu),
        ('model', LogisticRegression())
    ])
    params = {
        'model__random_state':range(0,x)
    }
    gs = GridSearchCV(pipe, params, verbose=1, n_jobs=-1, cv=2)
    
    #start timer
    start = datetime.now()
    
    gs.fit(df.nums.values.reshape(-1,1), df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_pf.append(diff.total_seconds())
    
eff_orig['pf_orig'] = eff_pf
```


```python
eff_cv = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('pre-process', CountVectorizer())
    ])
    pipe = Pipeline([
        ('fu', fu),
        ('model', LogisticRegression())
    ])
    params = {
        'model__random_state':range(0,x)
    }
    gs = GridSearchCV(pipe, params, verbose=1, n_jobs=-1, cv=2)
    
    #start timer
    start = datetime.now()
    
    gs.fit(df.text.values, df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_cv.append(diff.total_seconds())
    
eff_orig['cv_orig'] = eff_cv
```


```python
eff_tv = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('pre-process', TfidfVectorizer())
    ])
    pipe = Pipeline([
        ('fu', fu),
        ('model', LogisticRegression())
    ])
    params = {
        'model__random_state':range(0,x)
    }
    gs = GridSearchCV(pipe, params, verbose=1, n_jobs=-1, cv=2)
    
    #start timer
    start = datetime.now()
    
    gs.fit(df.text.values, df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_tv.append(diff.total_seconds())
    
eff_orig['tv_orig'] = eff_tv
```


```python
eff_cv_td = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('text', Pipeline([
            ('pre-process', CountVectorizer()),
            ('truncate', TruncatedSVD(n_components=1, random_state=rs))
        ]))
    ])
    pipe = Pipeline([
        ('fu', fu),
        ('model', LogisticRegression())
    ])
    params = {
        'model__random_state':range(0,x)
    }
    gs = GridSearchCV(pipe, params, verbose=1, n_jobs=-1, cv=2)
    
    #start timer
    start = datetime.now()
    
    gs.fit(df.text.values, df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_cv_td.append(diff.total_seconds())
    
eff_orig['cv_td_orig'] = eff_cv_td
```


```python
eff_tv_td = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('text', Pipeline([
            ('pre-process', TfidfVectorizer()),
            ('truncate', TruncatedSVD(n_components=1, random_state=rs))
        ]))
    ])
    pipe = Pipeline([
        ('fu', fu),
        ('model', LogisticRegression())
    ])
    params = {
        'model__random_state':range(0,x)
    }
    gs = GridSearchCV(pipe, params, verbose=1, n_jobs=-1, cv=2)
    
    #start timer
    start = datetime.now()
    
    gs.fit(df.text.values, df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_tv_td.append(diff.total_seconds())
    
eff_orig['tv_td_orig'] = eff_tv_td
```

Save Eff table to csv so no need to rerun


```python
eff_orig
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reps</th>
      <th>fits</th>
      <th>ss_orig</th>
      <th>pf_orig</th>
      <th>cv_orig</th>
      <th>tv_orig</th>
      <th>cv_td_orig</th>
      <th>tv_td_orig</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>0.225094</td>
      <td>0.210267</td>
      <td>6.232115</td>
      <td>4.133495</td>
      <td>4.286770</td>
      <td>4.524483</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
      <td>0.194651</td>
      <td>0.201753</td>
      <td>11.817011</td>
      <td>8.571318</td>
      <td>8.815194</td>
      <td>9.548219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>20</td>
      <td>0.207121</td>
      <td>0.206395</td>
      <td>18.046697</td>
      <td>13.498752</td>
      <td>13.731576</td>
      <td>15.187596</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>50</td>
      <td>0.310750</td>
      <td>0.312342</td>
      <td>39.263028</td>
      <td>31.436065</td>
      <td>31.830212</td>
      <td>32.558791</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>100</td>
      <td>0.424310</td>
      <td>0.407476</td>
      <td>73.355245</td>
      <td>58.983665</td>
      <td>59.402505</td>
      <td>60.533114</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75</td>
      <td>150</td>
      <td>0.521270</td>
      <td>0.522610</td>
      <td>106.176211</td>
      <td>86.964126</td>
      <td>88.038196</td>
      <td>89.465739</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>200</td>
      <td>0.527857</td>
      <td>0.547297</td>
      <td>139.342304</td>
      <td>115.449600</td>
      <td>116.504539</td>
      <td>118.238184</td>
    </tr>
    <tr>
      <th>7</th>
      <td>150</td>
      <td>300</td>
      <td>0.715433</td>
      <td>0.746119</td>
      <td>207.939189</td>
      <td>172.870768</td>
      <td>174.088018</td>
      <td>175.437701</td>
    </tr>
    <tr>
      <th>8</th>
      <td>200</td>
      <td>400</td>
      <td>0.841306</td>
      <td>0.806791</td>
      <td>274.387210</td>
      <td>228.160459</td>
      <td>230.957761</td>
      <td>233.329937</td>
    </tr>
    <tr>
      <th>9</th>
      <td>250</td>
      <td>500</td>
      <td>0.994794</td>
      <td>0.991691</td>
      <td>341.709382</td>
      <td>285.126249</td>
      <td>288.773181</td>
      <td>292.719185</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500</td>
      <td>1000</td>
      <td>1.674138</td>
      <td>1.710237</td>
      <td>678.446888</td>
      <td>568.165274</td>
      <td>573.782917</td>
      <td>579.152579</td>
    </tr>
    <tr>
      <th>11</th>
      <td>750</td>
      <td>1500</td>
      <td>2.371095</td>
      <td>2.430035</td>
      <td>1014.473684</td>
      <td>851.777699</td>
      <td>861.632572</td>
      <td>865.554674</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1000</td>
      <td>2000</td>
      <td>3.139456</td>
      <td>3.147182</td>
      <td>1352.139445</td>
      <td>1134.881215</td>
      <td>1146.537355</td>
      <td>1154.976613</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2500</td>
      <td>5000</td>
      <td>7.991735</td>
      <td>8.531802</td>
      <td>3367.759985</td>
      <td>2829.689819</td>
      <td>2864.428314</td>
      <td>2883.357771</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5000</td>
      <td>10000</td>
      <td>16.730484</td>
      <td>15.057567</td>
      <td>6756.219261</td>
      <td>5671.877889</td>
      <td>5720.519433</td>
      <td>5791.198268</td>
    </tr>
  </tbody>
</table>
</div>




```python
eff_orig.to_csv('efficiencyDForig.csv', index=False)
```

#### Try with Pipeline Memory Caching


```python
from tempfile import mkdtemp
from sklearn.externals.joblib import Memory
```


```python
cachedir = mkdtemp()
```


```python
eff_ss = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('pre-process', StandardScaler())
    ])
    pipe = Pipeline([
        ('fu', fu),
        ('model', LogisticRegression())
    ], memory=Memory(cachedir=cachedir, verbose=0))
    params = {
        'model__random_state':range(0,x)
    }
    gs = GridSearchCV(pipe, params, verbose=1, n_jobs=-1, cv=2)
    
    #start timer
    start = datetime.now()
    
    gs.fit(df.nums.values.reshape(-1,1), df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_ss.append(diff.total_seconds())
    
eff_cached['ss_cached'] = eff_ss
```


```python
eff_pf = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('pre-process', PolynomialFeatures())
    ])
    pipe = Pipeline([
        ('fu', fu),
        ('model', LogisticRegression())
    ], memory=Memory(cachedir=cachedir, verbose=0))
    params = {
        'model__random_state':range(0,x)
    }
    gs = GridSearchCV(pipe, params, verbose=1, n_jobs=-1, cv=2)
    
    #start timer
    start = datetime.now()
    
    gs.fit(df.nums.values.reshape(-1,1), df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_pf.append(diff.total_seconds())
    
eff_cached['pf_cached'] = eff_pf
```


```python
eff_cv = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('pre-process', CountVectorizer())
    ])
    pipe = Pipeline([
        ('fu', fu),
        ('model', LogisticRegression())
    ], memory=Memory(cachedir=cachedir, verbose=0))
    params = {
        'model__random_state':range(0,x)
    }
    gs = GridSearchCV(pipe, params, verbose=1, n_jobs=-1, cv=2)
    
    #start timer
    start = datetime.now()
    
    gs.fit(df.text.values, df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_cv.append(diff.total_seconds())
    
eff_cached['cv_cached'] = eff_cv
```


```python
eff_tv = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('pre-process', TfidfVectorizer())
    ])
    pipe = Pipeline([
        ('fu', fu),
        ('model', LogisticRegression())
    ], memory=Memory(cachedir=cachedir, verbose=0))
    params = {
        'model__random_state':range(0,x)
    }
    gs = GridSearchCV(pipe, params, verbose=1, n_jobs=-1, cv=2)
    
    #start timer
    start = datetime.now()
    
    gs.fit(df.text.values, df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_tv.append(diff.total_seconds())
    
eff_cached['tv_cached'] = eff_tv
```


```python
eff_cv_td = []http://localhost:8888/notebooks/PipelineEfficiency.ipynb#
for x in reps:
    #setup
    fu = FeatureUnion([
        ('text', Pipeline([
            ('pre-process', CountVectorizer()),
            ('truncate', TruncatedSVD(n_components=1, random_state=rs))
        ]))
    ])
    pipe = Pipeline([
        ('fu', fu),
        ('model', LogisticRegression())
    ], memory=Memory(cachedir=cachedir, verbose=0))
    params = {
        'model__random_state':range(0,x)
    }
    gs = GridSearchCV(pipe, params, verbose=1, n_jobs=-1, cv=2)
    
    #start timer
    start = datetime.now()
    
    gs.fit(df.text.values, df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_cv_td.append(diff.total_seconds())
    
eff_cached['cv_td_cached'] = eff_cv_td
```


```python
eff_tv_td = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('text', Pipeline([
            ('pre-process', TfidfVectorizer()),
            ('truncate', TruncatedSVD(n_components=1, random_state=rs))
        ]))
    ])
    pipe = Pipeline([
        ('fu', fu),
        ('model', LogisticRegression())
    ], memory=Memory(cachedir=cachedir, verbose=0))
    params = {
        'model__random_state':range(0,x)
    }
    gs = GridSearchCV(pipe, params, verbose=1, n_jobs=-1, cv=2)
    
    #start timer
    start = datetime.now()
    
    gs.fit(df.text.values, df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_tv_td.append(diff.total_seconds())
    
eff_cached['tv_td_cached'] = eff_tv_td
```


```python
eff_cached
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reps</th>
      <th>fits</th>
      <th>ss_cached</th>
      <th>pf_cached</th>
      <th>cv_cached</th>
      <th>tv_cached</th>
      <th>cv_td_cached</th>
      <th>tv_td_cached</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>0.220370</td>
      <td>0.213159</td>
      <td>8.577361</td>
      <td>6.538034</td>
      <td>6.739061</td>
      <td>6.718483</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
      <td>0.210165</td>
      <td>0.217480</td>
      <td>10.221625</td>
      <td>7.189253</td>
      <td>6.298098</td>
      <td>6.342444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>20</td>
      <td>0.211886</td>
      <td>0.215039</td>
      <td>15.564601</td>
      <td>11.348357</td>
      <td>10.216127</td>
      <td>10.532659</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>50</td>
      <td>0.326096</td>
      <td>0.325405</td>
      <td>34.597346</td>
      <td>26.822049</td>
      <td>24.723027</td>
      <td>25.100113</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>100</td>
      <td>0.417474</td>
      <td>0.424614</td>
      <td>64.362823</td>
      <td>50.839350</td>
      <td>47.331228</td>
      <td>48.007842</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75</td>
      <td>150</td>
      <td>0.598978</td>
      <td>0.532198</td>
      <td>94.691714</td>
      <td>75.714144</td>
      <td>70.753043</td>
      <td>71.175799</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>200</td>
      <td>0.657977</td>
      <td>0.604862</td>
      <td>125.989611</td>
      <td>100.350146</td>
      <td>93.655833</td>
      <td>94.807464</td>
    </tr>
    <tr>
      <th>7</th>
      <td>150</td>
      <td>300</td>
      <td>0.848278</td>
      <td>0.831244</td>
      <td>187.994630</td>
      <td>150.029235</td>
      <td>139.665444</td>
      <td>141.309259</td>
    </tr>
    <tr>
      <th>8</th>
      <td>200</td>
      <td>400</td>
      <td>1.000400</td>
      <td>1.069972</td>
      <td>249.437291</td>
      <td>199.660035</td>
      <td>186.383742</td>
      <td>188.557826</td>
    </tr>
    <tr>
      <th>9</th>
      <td>250</td>
      <td>500</td>
      <td>1.278369</td>
      <td>1.291126</td>
      <td>310.134293</td>
      <td>249.871598</td>
      <td>232.914126</td>
      <td>235.549512</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500</td>
      <td>1000</td>
      <td>2.243304</td>
      <td>2.248786</td>
      <td>616.757385</td>
      <td>499.087750</td>
      <td>464.231040</td>
      <td>468.925885</td>
    </tr>
    <tr>
      <th>11</th>
      <td>750</td>
      <td>1500</td>
      <td>3.232118</td>
      <td>3.320747</td>
      <td>919.324976</td>
      <td>745.893859</td>
      <td>696.744014</td>
      <td>702.962536</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1000</td>
      <td>2000</td>
      <td>4.202490</td>
      <td>4.257102</td>
      <td>1233.318759</td>
      <td>992.546725</td>
      <td>932.418590</td>
      <td>937.363592</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2500</td>
      <td>5000</td>
      <td>10.107030</td>
      <td>10.359197</td>
      <td>3063.400708</td>
      <td>2481.567094</td>
      <td>2326.713119</td>
      <td>2337.549000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5000</td>
      <td>10000</td>
      <td>19.715110</td>
      <td>20.546900</td>
      <td>6141.560259</td>
      <td>4969.542471</td>
      <td>4642.505104</td>
      <td>4678.323244</td>
    </tr>
  </tbody>
</table>
</div>




```python
eff_cached.to_csv('efficiencyDFcached.csv', index=False)
```

**Visualize Times**


```python
eff_orig = pd.read_csv('efficiencyDForig.csv')
eff_orig
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reps</th>
      <th>fits</th>
      <th>ss_orig</th>
      <th>pf_orig</th>
      <th>cv_orig</th>
      <th>tv_orig</th>
      <th>cv_td_orig</th>
      <th>tv_td_orig</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>0.225094</td>
      <td>0.210267</td>
      <td>6.232115</td>
      <td>4.133495</td>
      <td>4.286770</td>
      <td>4.524483</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
      <td>0.194651</td>
      <td>0.201753</td>
      <td>11.817011</td>
      <td>8.571318</td>
      <td>8.815194</td>
      <td>9.548219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>20</td>
      <td>0.207121</td>
      <td>0.206395</td>
      <td>18.046697</td>
      <td>13.498752</td>
      <td>13.731576</td>
      <td>15.187596</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>50</td>
      <td>0.310750</td>
      <td>0.312342</td>
      <td>39.263028</td>
      <td>31.436065</td>
      <td>31.830212</td>
      <td>32.558791</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>100</td>
      <td>0.424310</td>
      <td>0.407476</td>
      <td>73.355245</td>
      <td>58.983665</td>
      <td>59.402505</td>
      <td>60.533114</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75</td>
      <td>150</td>
      <td>0.521270</td>
      <td>0.522610</td>
      <td>106.176211</td>
      <td>86.964126</td>
      <td>88.038196</td>
      <td>89.465739</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>200</td>
      <td>0.527857</td>
      <td>0.547297</td>
      <td>139.342304</td>
      <td>115.449600</td>
      <td>116.504539</td>
      <td>118.238184</td>
    </tr>
    <tr>
      <th>7</th>
      <td>150</td>
      <td>300</td>
      <td>0.715433</td>
      <td>0.746119</td>
      <td>207.939189</td>
      <td>172.870768</td>
      <td>174.088018</td>
      <td>175.437701</td>
    </tr>
    <tr>
      <th>8</th>
      <td>200</td>
      <td>400</td>
      <td>0.841306</td>
      <td>0.806791</td>
      <td>274.387210</td>
      <td>228.160459</td>
      <td>230.957761</td>
      <td>233.329937</td>
    </tr>
    <tr>
      <th>9</th>
      <td>250</td>
      <td>500</td>
      <td>0.994794</td>
      <td>0.991691</td>
      <td>341.709382</td>
      <td>285.126249</td>
      <td>288.773181</td>
      <td>292.719185</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500</td>
      <td>1000</td>
      <td>1.674138</td>
      <td>1.710237</td>
      <td>678.446888</td>
      <td>568.165274</td>
      <td>573.782917</td>
      <td>579.152579</td>
    </tr>
    <tr>
      <th>11</th>
      <td>750</td>
      <td>1500</td>
      <td>2.371095</td>
      <td>2.430035</td>
      <td>1014.473684</td>
      <td>851.777699</td>
      <td>861.632572</td>
      <td>865.554674</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1000</td>
      <td>2000</td>
      <td>3.139456</td>
      <td>3.147182</td>
      <td>1352.139445</td>
      <td>1134.881215</td>
      <td>1146.537355</td>
      <td>1154.976613</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2500</td>
      <td>5000</td>
      <td>7.991735</td>
      <td>8.531802</td>
      <td>3367.759985</td>
      <td>2829.689819</td>
      <td>2864.428314</td>
      <td>2883.357771</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5000</td>
      <td>10000</td>
      <td>16.730484</td>
      <td>15.057567</td>
      <td>6756.219261</td>
      <td>5671.877889</td>
      <td>5720.519433</td>
      <td>5791.198268</td>
    </tr>
  </tbody>
</table>
</div>




```python
eff_cached = pd.read_csv('efficiencyDFcached.csv')
eff_cached
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reps</th>
      <th>fits</th>
      <th>ss_cached</th>
      <th>pf_cached</th>
      <th>cv_cached</th>
      <th>tv_cached</th>
      <th>cv_td_cached</th>
      <th>tv_td_cached</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>0.220370</td>
      <td>0.213159</td>
      <td>8.577361</td>
      <td>6.538034</td>
      <td>6.739061</td>
      <td>6.718483</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
      <td>0.210165</td>
      <td>0.217480</td>
      <td>10.221625</td>
      <td>7.189253</td>
      <td>6.298098</td>
      <td>6.342444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>20</td>
      <td>0.211886</td>
      <td>0.215039</td>
      <td>15.564601</td>
      <td>11.348357</td>
      <td>10.216127</td>
      <td>10.532659</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>50</td>
      <td>0.326096</td>
      <td>0.325405</td>
      <td>34.597346</td>
      <td>26.822049</td>
      <td>24.723027</td>
      <td>25.100113</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>100</td>
      <td>0.417474</td>
      <td>0.424614</td>
      <td>64.362823</td>
      <td>50.839350</td>
      <td>47.331228</td>
      <td>48.007842</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75</td>
      <td>150</td>
      <td>0.598978</td>
      <td>0.532198</td>
      <td>94.691714</td>
      <td>75.714144</td>
      <td>70.753043</td>
      <td>71.175799</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>200</td>
      <td>0.657977</td>
      <td>0.604862</td>
      <td>125.989611</td>
      <td>100.350146</td>
      <td>93.655833</td>
      <td>94.807464</td>
    </tr>
    <tr>
      <th>7</th>
      <td>150</td>
      <td>300</td>
      <td>0.848278</td>
      <td>0.831244</td>
      <td>187.994630</td>
      <td>150.029235</td>
      <td>139.665444</td>
      <td>141.309259</td>
    </tr>
    <tr>
      <th>8</th>
      <td>200</td>
      <td>400</td>
      <td>1.000400</td>
      <td>1.069972</td>
      <td>249.437291</td>
      <td>199.660035</td>
      <td>186.383742</td>
      <td>188.557826</td>
    </tr>
    <tr>
      <th>9</th>
      <td>250</td>
      <td>500</td>
      <td>1.278369</td>
      <td>1.291126</td>
      <td>310.134293</td>
      <td>249.871598</td>
      <td>232.914126</td>
      <td>235.549512</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500</td>
      <td>1000</td>
      <td>2.243304</td>
      <td>2.248786</td>
      <td>616.757385</td>
      <td>499.087750</td>
      <td>464.231040</td>
      <td>468.925885</td>
    </tr>
    <tr>
      <th>11</th>
      <td>750</td>
      <td>1500</td>
      <td>3.232118</td>
      <td>3.320747</td>
      <td>919.324976</td>
      <td>745.893859</td>
      <td>696.744014</td>
      <td>702.962536</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1000</td>
      <td>2000</td>
      <td>4.202490</td>
      <td>4.257102</td>
      <td>1233.318759</td>
      <td>992.546725</td>
      <td>932.418590</td>
      <td>937.363592</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2500</td>
      <td>5000</td>
      <td>10.107030</td>
      <td>10.359197</td>
      <td>3063.400708</td>
      <td>2481.567094</td>
      <td>2326.713119</td>
      <td>2337.549000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5000</td>
      <td>10000</td>
      <td>19.715110</td>
      <td>20.546900</td>
      <td>6141.560259</td>
      <td>4969.542471</td>
      <td>4642.505104</td>
      <td>4678.323244</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(16,8))
plt.plot(eff_orig['fits'], eff_orig['ss_orig'], 
         lw=3, color='red', label='Standard Scaler Original')
plt.plot(eff_orig['fits'], eff_orig['pf_orig'], 
         lw=3, color='red', linestyle='--', label='Polynomial Feature Original')
plt.plot(eff_orig['fits'], eff_orig['cv_orig'], 
         lw=3, color='red', linestyle='-.', label='Count Vectorizer Original')
plt.plot(eff_orig['fits'], eff_orig['tv_orig'], 
         lw=3, color='red', linestyle=':', label='TfidVectorizer Original')

plt.plot(eff_cached['fits'], eff_cached['ss_cached'], 
         lw=3, color='blue', label='Standard Scaler Memory Cached')
plt.plot(eff_cached['fits'], eff_cached['pf_cached'], 
         lw=3, color='blue', linestyle='--', label='Polynomial Feature Memory Cached')
plt.plot(eff_cached['fits'], eff_cached['cv_cached'], 
         lw=3, color='blue', linestyle='-.', label='Count Vectorizer Memory Cached')
plt.plot(eff_cached['fits'], eff_cached['tv_cached'], 
         lw=3, color='blue', linestyle=':', label='TfidVectorizer Memory Cached')

plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Pipelines")
plt.legend(loc = 'best')
plt.show()
```


![png](PipelineEfficiency_files/PipelineEfficiency_32_0.png)



```python
fig, ax = plt.subplots(figsize=(16,8))
plt.plot(eff_orig['fits'], eff_orig['ss_orig'], 
         lw=3, color='red', label='Standard Scaler Original')
plt.plot(eff_orig['fits'], eff_orig['pf_orig'], 
         lw=3, color='red', linestyle='--', label='Polynomial Feature Original')

plt.plot(eff_cached['fits'], eff_cached['ss_cached'], 
         lw=3, color='blue', label='Standard Scaler Memory Cached')
plt.plot(eff_cached['fits'], eff_cached['pf_cached'], 
         lw=3, color='blue', linestyle='--', label='Polynomial Feature Memory Cached')

plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Pipelines")
plt.legend(loc = 'best')
plt.show()
```


![png](PipelineEfficiency_files/PipelineEfficiency_33_0.png)



```python
fig, ax = plt.subplots(figsize=(16,8))
plt.plot(eff_orig['fits'], eff_orig['cv_orig'], 
         lw=3, color='red', linestyle='-', label='Count Vectorizer Original')
plt.plot(eff_orig['fits'], eff_orig['tv_orig'], 
         lw=3, color='red', linestyle='--', label='TfidVectorizer Original')

plt.plot(eff_cached['fits'], eff_cached['cv_cached'], 
         lw=3, color='blue', linestyle='-', label='Count Vectorizer Memory Cached')
plt.plot(eff_cached['fits'], eff_cached['tv_cached'], 
         lw=3, color='blue', linestyle='--', label='TfidVectorizer Memory Cached')

plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Pipelines")
plt.legend(loc = 'best')
plt.show()
```


![png](PipelineEfficiency_files/PipelineEfficiency_34_0.png)



```python
fig = plt.figure(figsize=(20,25))

ax1 = fig.add_subplot(3,2,1)
ax1.plot(eff_orig['fits'], eff_orig['ss_orig'], 
         lw=3, color='red', label='Original')
ax1.plot(eff_cached['fits'], eff_cached['ss_cached'], 
         lw=3, color='blue', label='Memory Cached')
plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Standard Scalar Pipeline")
plt.legend(loc = 'best')
plt.grid()

ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(eff_orig['fits'], eff_orig['pf_orig'], 
         lw=3, color='red', label='Original')
ax2.plot(eff_cached['fits'], eff_cached['pf_cached'], 
         lw=3, color='blue', label='Memory Cached')
plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Polynomial Feature Pipeline")
plt.legend(loc = 'best')
plt.grid()

ax3 = fig.add_subplot(3, 2, 3)
ax3.plot(eff_orig['fits'], eff_orig['cv_orig'], 
         lw=3, color='red', label='Original')
ax3.plot(eff_cached['fits'], eff_cached['cv_cached'], 
         lw=3, color='blue', label='Memory Cached')
plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Count Vectorizer Pipeline")
plt.legend(loc = 'best')
plt.grid()

ax4 = fig.add_subplot(3, 2, 4)
ax4.plot(eff_orig['fits'], eff_orig['tv_orig'], 
         lw=3, color='red', label='Original')
ax4.plot(eff_cached['fits'], eff_cached['tv_cached'], 
         lw=3, color='blue', label='Memory Cached')
plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit TfidVectorizer Pipeline")
plt.legend(loc = 'best')
plt.grid()

ax5 = fig.add_subplot(3, 2, 5)
ax5.plot(eff_orig['fits'], eff_orig['cv_td_orig'], 
         lw=3, color='red', label='Original')
ax5.plot(eff_cached['fits'], eff_cached['cv_td_cached'], 
         lw=3, color='blue', label='Memory Cached')
plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Count Vectorizer with Truncation Pipeline")
plt.legend(loc = 'best')
plt.grid()

ax6 = fig.add_subplot(3, 2, 6)
ax6.plot(eff_orig['fits'], eff_orig['tv_td_orig'], 
         lw=3, color='red', label='Original')
ax6.plot(eff_cached['fits'], eff_cached['tv_td_cached'], 
         lw=3, color='blue', label='Memory Cached')
plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to TfidVectorizor with Truncation Pipeline")
plt.legend(loc = 'best')
plt.grid()

plt.show()
```


![png](PipelineEfficiency_files/PipelineEfficiency_35_0.png)


### Solving the Issue


```python
import pprint
pp = pprint.PrettyPrinter()
```


```python
text_pipe = Pipeline([('nlp', CountVectorizer())])

fu = FeatureUnion([
                   ('text', text_pipe),
                   ('text2', text_pipe),
                   ('BLAHBLAH', text_pipe)
                  ])

modeling_pipe = Pipeline([
                            ('data', fu),
                            ('model', LogisticRegression())
                        ])
```


```python
def logan_search_fit_helper(pipe, params, fset):
    pipe.set_params(**params)
    scores = []
    for k in fset:
        #k[0]=Xtr, k[1]=Xte, k[2]=ytr, k[3]=yte
        if isinstance(k[0], np.ndarray):
            #Is a numpy array
            pipe.fit(k[0], k[2])
            scores.append(pipe.score(k[1], k[3]))
        else:
            #Is not Numpy Array
            pipe.fit(k[0].copy(), k[2].copy())
            scores.append(pipe.score(k[1].copy(), k[3].copy()))
    #
    return (np.mean(scores), params)
```


```python
class LoganSearch(BaseSearchCV):
    
    def __init__(self, fu, estimator, param_grid, fu_params={}, 
                 n_jobs=1, cv=3, verbose=0):
        #set unique attributes
        self._fus_ = []
        self._base_fu = fu
        self._base_fu_params = fu_params
        self.cv_ = cv
        #set attributes for results
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        #set attributes for GridSearch
        self._estimator = estimator
        self._param_grid = param_grid
        self._n_jobs = n_jobs
        self._verbose = verbose
        
    def fit(self, X, y):
        self.__set_all_fus(self._base_fu, self._base_fu_params)
        folder = check_cv(cv=self.cv_, y=y, classifier=is_classifier(self._estimator))
        scores = []
        for fu in self._fus_:
            fset = []
            for tr, te in folder.split(X,y):
                Xtmp_tr = fu.fit_transform(X[tr])
                Xtmp_te = fu.transform(X[te])
                ytmp_tr = y[tr]
                ytmp_te = y[te]
                fset.append((Xtmp_tr.copy(), Xtmp_te.copy(), 
                             ytmp_tr.copy(), ytmp_te.copy()))
            #
            print('Done Transforming Data')
            print("----------------------------")
            n_splits = folder.get_n_splits()
            n_candidates = len(ParameterGrid(self._param_grid))
            if self._verbose > 0:
                print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))

            tmp_results = Parallel(n_jobs=self._n_jobs, verbose=self._verbose
                                  )(delayed(logan_search_fit_helper)
                                    (clone(self._estimator), params, fset) 
                                    for params in ParameterGrid(self._param_grid))
            scores.extend(tmp_results)
            #
        #
        scores.sort(reverse=True, key=lambda x: x[0])
        self.best_score_ = scores[0][0]
        self.best_params_ = scores[0][1]
        return None
    
    def score(self, X, y):
        if not self.best_estimator_:
            print('Model must first be fit')
            return None
        else:
            return self.best_estimator_.score(X, y)
    
    def predict(self, X, y):
        if not self.best_estimator_:
            print('Model must first be fit')
            return None
        else:
            return self.best_estimator_.predict(X, y)
    
    def predict_proba(self, X, y):
        if not self.best_estimator_:
            print('Model must first be fit')
            return None
        else:
            return self.best_estimator_.predict_proba(X, y)
    
    def __set_all_fus(self, base_fu, fu_params):
        tmp_fus = []
        for p_name, pipe in base_fu.transformer_list:
            tmp_params = sorted([k for k in fu_params.keys() if (k.split("__")[0]==p_name)], 
                                key=lambda x: x.count('__'))
            #are there parameters to change for this pipe?
            #there are no parameters matching this pipe, so add as is
            if not tmp_params: 
                #list of feature unions is empty
                if not tmp_fus: 
                    tmp_fus.append(FeatureUnion([(p_name, pipe)]))
                #already feature unions, append to them
                else: 
                    for tmp_fu in tmp_fus:
                        tmp_fu.transformer_list.append((p_name, pipe))
            #there are paramters to change   
            else:
                tmp_pipes = [pipe]
                if tmp_params[0] == p_name:
                    tmp_pipes = fu_params[p_name]
#                     tmp_pipes = [self.add_params(tmp_pipe, {p_name: tmp_pipe}) 
#                                  for tmp_pipe in fu_params[p_name]]
                    tmp_params = tmp_params[1:]
                #
                new_pipes = []
                if tmp_params:
                    for pipe in tmp_pipes:
                        param_dict = {k:v for k, v in fu_params.items() if k in tmp_params}
                        for param_comb in ParameterGrid(param_dict):
                            tmp_pipe = clone(pipe)
                            #tmp_pipe = self.add_params(tmp_pipe, param_comb)
                            params_edited = {k.lstrip(p_name+"__"):v for k,v in param_comb.items()}
                            tmp_pipe.set_params(**params_edited)
                            new_pipes.append(clone(tmp_pipe))
                #
                #
                else:
                    new_pipes = tmp_pipes

                #add new pipes to feature unions
                new_fus = []
                #list of feature unions is empty
                if not tmp_fus: 
                    for pipe in new_pipes:
                            new_fu = FeatureUnion([(p_name, pipe)])
                            new_fus.append(new_fu)
                #already feature unions, append to them
                else:
                    for tmp_fu in tmp_fus:
                        for pipe in new_pipes:
                            new_fu = clone(tmp_fu)
                            new_fu.transformer_list.append((p_name, pipe))
                            new_fus.append(clone(new_fu))
                #        
                tmp_fus = new_fus
            #
        #
        self._fus_ = tmp_fus
        return None
    
    def add_params(obj, params):
        if hasattr(obj, "params"):
            obj.params.update(params)
        else:
            obj.params = params
        return obj
    
```


```python
text_pipe = Pipeline([('nlp', CountVectorizer())])

fu = FeatureUnion([
                   ('text', text_pipe),
                   ('text2', text_pipe),
                   ('BLAHBLAH', text_pipe)
                  ])

modeling_pipe = Pipeline([
                            ('data', fu),
                            ('model', LogisticRegression())
                        ])
```


```python
params = {
    'model':[LogisticRegression(penalty='l1'),
            LogisticRegression(penalty='l2')]
}

gs = GridSearchCV(modeling_pipe, params, cv=3, n_jobs=-1, verbose=1)

fu_params1 = {
    'text__nlp': [CountVectorizer(),
                      CountVectorizer(ngram_range=(1,2)),
                      TfidfVectorizer(ngram_range=(1,2)),
                      TfidfVectorizer()]
}

fu_params2 = {
    'text': [Pipeline([('nlp2a', CountVectorizer())]),
            Pipeline([('nlp2b', TfidfVectorizer())])
            ]
}

fu_params3 = {
    'text__nlp': [CountVectorizer(),
                    TfidfVectorizer()],
    'text__nlp__ngram_range':[(1,1), (1,2)],
    'text2': [Pipeline([('nlp2a', CountVectorizer())]),
            Pipeline([('nlp2b', TfidfVectorizer())])
            ]
}
```


```python
tfu = FeatureUnion([
    ('text', StandardScaler())
])

tfpipe = Pipeline([
    ('model', LogisticRegression())
])

tpipe = Pipeline([
    ('data', tfu),
    ('model', LogisticRegression())
])

tparams = {
    'model__random_state':range(0,5000)
}

cv = 2

ls = LoganSearch(tfu, tfpipe, tparams, n_jobs=-1, cv=2, verbose=1)
ls.fit(df.nums.values.reshape(-1,1), df.relevance)
pp.pprint((ls.best_score_, ls.best_params_))


gs = GridSearchCV(tpipe, tparams, cv=cv, n_jobs=-1, verbose=1)
gs.fit(df.nums.values.reshape(-1,1), df.relevance)
pp.pprint((gs.best_score_, gs.best_params_))
```

    Fitting 2 folds for each of 5000 candidates, totalling 10000 fits


    [Parallel(n_jobs=-1)]: Done 232 tasks      | elapsed:    0.6s
    [Parallel(n_jobs=-1)]: Done 2032 tasks      | elapsed:    3.4s
    [Parallel(n_jobs=-1)]: Done 5000 out of 5000 | elapsed:    8.0s finished


    (0.82250000000000001, {'model__random_state': 0})
    Fitting 2 folds for each of 5000 candidates, totalling 10000 fits


    [Parallel(n_jobs=-1)]: Done 268 tasks      | elapsed:    0.7s
    [Parallel(n_jobs=-1)]: Done 2368 tasks      | elapsed:    3.6s
    [Parallel(n_jobs=-1)]: Done 5868 tasks      | elapsed:    8.5s


    (0.82250000000000001, {'model__random_state': 0})


    [Parallel(n_jobs=-1)]: Done 10000 out of 10000 | elapsed:   14.0s finished


#### Run Test with New Function

I will now run the same tests as previously performed, however, I will be using the new LoganSearch class instead of GridSearchCV, and the pipeline will be formatted slightly different to account for that, but will still have the same format.

Reset Params to make it easier to run only parts of the notebook


```python
df = pd.read_csv('sample_data/economic_news.csv', usecols=[7, 11, 14])
df.text = df.text.apply(lambda x: x.replace('</br>', ''))
df.relevance = df.relevance.apply(lambda x: 1 if x == 'yes' else 0)
print(df.shape)
df.head()
```

    (8000, 3)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>relevance</th>
      <th>headline</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Yields on CDs Fell in the Latest Week</td>
      <td>NEW YORK -- Yields on most certificates of dep...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>The Morning Brief: White House Seeks to Limit ...</td>
      <td>The Wall Street Journal OnlineThe Morning Brie...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Banking Bill Negotiators Set Compromise --- Pl...</td>
      <td>WASHINGTON -- In an effort to achieve banking ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Manager's Journal: Sniffing Out Drug Abusers I...</td>
      <td>The statistics on the enormous costs of employ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Currency Trading: Dollar Remains in Tight Rang...</td>
      <td>NEW YORK -- Indecision marked the dollar's ton...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#set up some stuff

#of times to GridSearch something
reps = [1,5,10,25,50,75,100,150,200,250,500,750,1000,2500,5000]

tmp = []
df['nums'] = df.index
rs = 779
np.random.seed(rs)
df.nums = df.nums.apply(lambda x: 100*x* np.random.rand())
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>relevance</th>
      <th>headline</th>
      <th>text</th>
      <th>nums</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Yields on CDs Fell in the Latest Week</td>
      <td>NEW YORK -- Yields on most certificates of dep...</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>The Morning Brief: White House Seeks to Limit ...</td>
      <td>The Wall Street Journal OnlineThe Morning Brie...</td>
      <td>72.259737</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Banking Bill Negotiators Set Compromise --- Pl...</td>
      <td>WASHINGTON -- In an effort to achieve banking ...</td>
      <td>184.913663</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Manager's Journal: Sniffing Out Drug Abusers I...</td>
      <td>The statistics on the enormous costs of employ...</td>
      <td>94.107106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Currency Trading: Dollar Remains in Tight Rang...</td>
      <td>NEW YORK -- Indecision marked the dollar's ton...</td>
      <td>336.160412</td>
    </tr>
  </tbody>
</table>
</div>




```python
eff_logan = pd.DataFrame(reps, columns=['reps'])
eff_logan['fits'] = eff_logan.reps.apply(lambda x: x*2)
eff_logan
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reps</th>
      <th>fits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>100</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75</td>
      <td>150</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>200</td>
    </tr>
    <tr>
      <th>7</th>
      <td>150</td>
      <td>300</td>
    </tr>
    <tr>
      <th>8</th>
      <td>200</td>
      <td>400</td>
    </tr>
    <tr>
      <th>9</th>
      <td>250</td>
      <td>500</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>750</td>
      <td>1500</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1000</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2500</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5000</td>
      <td>10000</td>
    </tr>
  </tbody>
</table>
</div>




```python
eff_ss = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('pre-process', StandardScaler())
    ])
    pipe = Pipeline([
        ('model', LogisticRegression())
    ])
    params = {
        'model__random_state':range(0,x)
    }
    ls = LoganSearch(fu, pipe, params, n_jobs=-1, cv=2, verbose=1)
    
    #start timer
    start = datetime.now()
    
    ls.fit(df.nums.values.reshape(-1,1), df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_ss.append(diff.total_seconds())
    
eff_logan['ss_logan'] = eff_ss
```


```python
eff_pf = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('pre-process', PolynomialFeatures())
    ])
    pipe = Pipeline([
        ('model', LogisticRegression())
    ])
    params = {
        'model__random_state':range(0,x)
    }
    ls = LoganSearch(fu, pipe, params, n_jobs=-1, cv=2, verbose=1)
    
    #start timer
    start = datetime.now()
    
    ls.fit(df.nums.values.reshape(-1,1), df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_pf.append(diff.total_seconds())
    
eff_logan['pf_logan'] = eff_pf
```


```python
eff_cv = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('pre-process', CountVectorizer())
    ])
    pipe = Pipeline([
        ('model', LogisticRegression())
    ])
    params = {
        'model__random_state':range(0,x)
    }
    ls = LoganSearch(fu, pipe, params, n_jobs=-1, cv=2, verbose=1)
    
    #start timer
    start = datetime.now()
    
    ls.fit(df.text.values, df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_cv.append(diff.total_seconds())
    
eff_logan['cv_logan'] = eff_cv
```

    Done Transforming Data
    ----------------------------
    Fitting 2 folds for each of 1 candidates, totalling 2 fits


    [Parallel(n_jobs=-1)]: Done   1 out of   1 | elapsed:    1.1s finished



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-144-a8aa73131bd1> in <module>()
         21     stop = datetime.now()
         22     diff = stop - start
    ---> 23     eff_tv.append(diff.total_seconds())
         24 
         25 eff_logan['cv_logan'] = eff_cv


    NameError: name 'eff_tv' is not defined



```python
eff_tv = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('pre-process', TfidfVectorizer())
    ])
    pipe = Pipeline([
        ('model', LogisticRegression())
    ])
    params = {
        'model__random_state':range(0,x)
    }
    ls = LoganSearch(fu, pipe, params, n_jobs=-1, cv=2, verbose=1)
    
    #start timer
    start = datetime.now()
    
    ls.fit(df.text.values, df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_tv.append(diff.total_seconds())
    
eff_logan['tv_logan'] = eff_tv
```


```python
eff_cv_td = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('text', Pipeline([
            ('pre-process', CountVectorizer()),
            ('truncate', TruncatedSVD(n_components=1, random_state=rs))
        ]))
    ])
    pipe = Pipeline([
        ('model', LogisticRegression())
    ])
    params = {
        'model__random_state':range(0,x)
    }
    ls = LoganSearch(fu, pipe, params, n_jobs=-1, cv=2, verbose=1)
    
    #start timer
    start = datetime.now()
    
    ls.fit(df.text.values, df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_cv_td.append(diff.total_seconds())
    
eff_logan['cv_td_logan'] = eff_cv_td
```


```python
eff_tv_td = []
for x in reps:
    #setup
    fu = FeatureUnion([
        ('text', Pipeline([
            ('pre-process', TfidfVectorizer()),
            ('truncate', TruncatedSVD(n_components=1, random_state=rs))
        ]))
    ])
    pipe = Pipeline([
        ('model', LogisticRegression())
    ])
    params = {
        'model__random_state':range(0,x)
    }
    ls = LoganSearch(fu, pipe, params, n_jobs=-1, cv=2, verbose=1)
    
    #start timer
    start = datetime.now()
    
    ls.fit(df.text.values, df.relevance)
    
    #end timer
    stop = datetime.now()
    diff = stop - start
    eff_tv_td.append(diff.total_seconds())
    
eff_logan['tv_td_logan'] = eff_tv_td
```

Save to file


```python
eff_logan
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reps</th>
      <th>fits</th>
      <th>ss_logan</th>
      <th>pf_logan</th>
      <th>cv_logan</th>
      <th>tv_logan</th>
      <th>cv_td_logan</th>
      <th>tv_td_logan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>0.223909</td>
      <td>0.221798</td>
      <td>6.232115</td>
      <td>3.607295</td>
      <td>3.185836</td>
      <td>3.321109</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
      <td>0.242797</td>
      <td>0.226656</td>
      <td>11.817011</td>
      <td>3.725415</td>
      <td>3.517862</td>
      <td>3.260362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>20</td>
      <td>0.224017</td>
      <td>0.232498</td>
      <td>18.046697</td>
      <td>4.194502</td>
      <td>3.550491</td>
      <td>3.297210</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>50</td>
      <td>0.331112</td>
      <td>0.346223</td>
      <td>39.263028</td>
      <td>5.359049</td>
      <td>3.521202</td>
      <td>3.375086</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>100</td>
      <td>0.334166</td>
      <td>0.456564</td>
      <td>73.355245</td>
      <td>7.684017</td>
      <td>3.552729</td>
      <td>3.456631</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75</td>
      <td>150</td>
      <td>0.331425</td>
      <td>0.460818</td>
      <td>106.176211</td>
      <td>9.456448</td>
      <td>3.465066</td>
      <td>3.469402</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>200</td>
      <td>0.433914</td>
      <td>0.566037</td>
      <td>139.342304</td>
      <td>11.695089</td>
      <td>3.420275</td>
      <td>3.470165</td>
    </tr>
    <tr>
      <th>7</th>
      <td>150</td>
      <td>300</td>
      <td>0.570316</td>
      <td>0.570319</td>
      <td>207.939189</td>
      <td>15.477273</td>
      <td>3.633281</td>
      <td>3.938534</td>
    </tr>
    <tr>
      <th>8</th>
      <td>200</td>
      <td>400</td>
      <td>0.630874</td>
      <td>0.773557</td>
      <td>274.387210</td>
      <td>18.273176</td>
      <td>3.749651</td>
      <td>3.728653</td>
    </tr>
    <tr>
      <th>9</th>
      <td>250</td>
      <td>500</td>
      <td>0.757504</td>
      <td>0.880124</td>
      <td>341.709382</td>
      <td>22.261672</td>
      <td>3.892925</td>
      <td>3.772603</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500</td>
      <td>1000</td>
      <td>1.172229</td>
      <td>1.387544</td>
      <td>678.446888</td>
      <td>37.435749</td>
      <td>4.684382</td>
      <td>4.733261</td>
    </tr>
    <tr>
      <th>11</th>
      <td>750</td>
      <td>1500</td>
      <td>1.525951</td>
      <td>1.821197</td>
      <td>1014.473684</td>
      <td>57.568446</td>
      <td>4.804530</td>
      <td>5.297128</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1000</td>
      <td>2000</td>
      <td>1.926059</td>
      <td>2.506788</td>
      <td>1352.139445</td>
      <td>79.162025</td>
      <td>5.350899</td>
      <td>5.342550</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2500</td>
      <td>5000</td>
      <td>4.827360</td>
      <td>5.549650</td>
      <td>3367.759985</td>
      <td>186.087186</td>
      <td>8.388797</td>
      <td>7.705426</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5000</td>
      <td>10000</td>
      <td>8.561794</td>
      <td>9.462130</td>
      <td>6756.219261</td>
      <td>356.825039</td>
      <td>14.033000</td>
      <td>12.286837</td>
    </tr>
  </tbody>
</table>
</div>




```python
eff_logan.to_csv('efficiencyDFlogan.csv', index=False)
```

## Visualization and Conclusion


```python
eff_orig = pd.read_csv('efficiencyDForig.csv')
eff_cached = pd.read_csv('efficiencyDFcached.csv')
eff_logan = pd.read_csv('efficiencyDFlogan.csv')
```


```python
eff_logan
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reps</th>
      <th>fits</th>
      <th>ss_logan</th>
      <th>pf_logan</th>
      <th>cv_logan</th>
      <th>tv_logan</th>
      <th>cv_td_logan</th>
      <th>tv_td_logan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>0.223909</td>
      <td>0.221798</td>
      <td>6.232115</td>
      <td>3.607295</td>
      <td>3.185836</td>
      <td>3.321109</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
      <td>0.242797</td>
      <td>0.226656</td>
      <td>11.817011</td>
      <td>3.725415</td>
      <td>3.517862</td>
      <td>3.260362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>20</td>
      <td>0.224017</td>
      <td>0.232498</td>
      <td>18.046697</td>
      <td>4.194502</td>
      <td>3.550491</td>
      <td>3.297210</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>50</td>
      <td>0.331112</td>
      <td>0.346223</td>
      <td>39.263028</td>
      <td>5.359049</td>
      <td>3.521202</td>
      <td>3.375086</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>100</td>
      <td>0.334166</td>
      <td>0.456564</td>
      <td>73.355245</td>
      <td>7.684017</td>
      <td>3.552729</td>
      <td>3.456631</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75</td>
      <td>150</td>
      <td>0.331425</td>
      <td>0.460818</td>
      <td>106.176211</td>
      <td>9.456448</td>
      <td>3.465066</td>
      <td>3.469402</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>200</td>
      <td>0.433914</td>
      <td>0.566037</td>
      <td>139.342304</td>
      <td>11.695089</td>
      <td>3.420275</td>
      <td>3.470165</td>
    </tr>
    <tr>
      <th>7</th>
      <td>150</td>
      <td>300</td>
      <td>0.570316</td>
      <td>0.570319</td>
      <td>207.939189</td>
      <td>15.477273</td>
      <td>3.633281</td>
      <td>3.938534</td>
    </tr>
    <tr>
      <th>8</th>
      <td>200</td>
      <td>400</td>
      <td>0.630874</td>
      <td>0.773557</td>
      <td>274.387210</td>
      <td>18.273176</td>
      <td>3.749651</td>
      <td>3.728653</td>
    </tr>
    <tr>
      <th>9</th>
      <td>250</td>
      <td>500</td>
      <td>0.757504</td>
      <td>0.880124</td>
      <td>341.709382</td>
      <td>22.261672</td>
      <td>3.892925</td>
      <td>3.772603</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500</td>
      <td>1000</td>
      <td>1.172229</td>
      <td>1.387544</td>
      <td>678.446888</td>
      <td>37.435749</td>
      <td>4.684382</td>
      <td>4.733261</td>
    </tr>
    <tr>
      <th>11</th>
      <td>750</td>
      <td>1500</td>
      <td>1.525951</td>
      <td>1.821197</td>
      <td>1014.473684</td>
      <td>57.568446</td>
      <td>4.804530</td>
      <td>5.297128</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1000</td>
      <td>2000</td>
      <td>1.926059</td>
      <td>2.506788</td>
      <td>1352.139445</td>
      <td>79.162025</td>
      <td>5.350899</td>
      <td>5.342550</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2500</td>
      <td>5000</td>
      <td>4.827360</td>
      <td>5.549650</td>
      <td>3367.759985</td>
      <td>186.087186</td>
      <td>8.388797</td>
      <td>7.705426</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5000</td>
      <td>10000</td>
      <td>8.561794</td>
      <td>9.462130</td>
      <td>6756.219261</td>
      <td>356.825039</td>
      <td>14.033000</td>
      <td>12.286837</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(16,8))
plt.plot(eff_orig['fits'], eff_orig['ss_orig'], 
         lw=2.5, color='red', label='Standard Scaler Original')
plt.plot(eff_orig['fits'], eff_orig['pf_orig'], 
         lw=2.5, color='red', linestyle='--', label='Polynomial Feature Original')
plt.plot(eff_orig['fits'], eff_orig['cv_orig'], 
         lw=2.5, color='red', linestyle='-.', label='Count Vectorizer Original')
plt.plot(eff_orig['fits'], eff_orig['tv_orig'], 
         lw=3, color='red', linestyle=':', label='TfidVectorizer Original')

plt.plot(eff_cached['fits'], eff_cached['ss_cached'], 
         lw=2.5, color='blue', label='Standard Scaler Memory Cached')
plt.plot(eff_cached['fits'], eff_cached['pf_cached'], 
         lw=2.5, color='blue', linestyle='--', label='Polynomial Feature Memory Cached')
plt.plot(eff_cached['fits'], eff_cached['cv_cached'], 
         lw=2.5, color='blue', linestyle='-.', label='Count Vectorizer Memory Cached')
plt.plot(eff_cached['fits'], eff_cached['tv_cached'], 
         lw=3, color='blue', linestyle=':', label='TfidVectorizer Memory Cached')

plt.plot(eff_logan['fits'], eff_logan['ss_logan'], 
         lw=2.5, color='black', label='Standard Scaler New')
plt.plot(eff_logan['fits'], eff_logan['pf_logan'], 
         lw=2.5, color='black', linestyle='--', label='Polynomial Feature New')
plt.plot(eff_logan['fits'], eff_logan['cv_logan'], 
         lw=2.5, color='black', linestyle='-.', label='Count Vectorizer New')
plt.plot(eff_logan['fits'], eff_logan['tv_logan'], 
         lw=3, color='black', linestyle=':', label='TfidVectorizer New')

plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Pipelines")
plt.legend(loc = 'best')
plt.show()
```


![png](PipelineEfficiency_files/PipelineEfficiency_61_0.png)



```python
fig, ax = plt.subplots(figsize=(16,8))
plt.plot(eff_orig['fits'], eff_orig['ss_orig'], 
         lw=2.5, color='red', label='Standard Scaler Original')
plt.plot(eff_orig['fits'], eff_orig['pf_orig'], 
         lw=2.5, color='red', linestyle='--', label='Polynomial Feature Original')

plt.plot(eff_cached['fits'], eff_cached['ss_cached'], 
         lw=2.5, color='blue', label='Standard Scaler Memory Cached')
plt.plot(eff_cached['fits'], eff_cached['pf_cached'], 
         lw=2.5, color='blue', linestyle='--', label='Polynomial Feature Memory Cached')

plt.plot(eff_logan['fits'], eff_logan['ss_logan'], 
         lw=2.5, color='black', label='Standard Scaler New')
plt.plot(eff_logan['fits'], eff_logan['pf_logan'], 
         lw=2.5, color='black', linestyle='--', label='Polynomial Feature New')

plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Pipelines")
plt.legend(loc = 'best')
plt.show()
```


![png](PipelineEfficiency_files/PipelineEfficiency_62_0.png)



```python
fig, ax = plt.subplots(figsize=(16,8))
plt.plot(eff_orig['fits'], eff_orig['cv_orig'], 
         lw=2.5, color='red', linestyle='-', label='Count Vectorizer Original')
plt.plot(eff_orig['fits'], eff_orig['tv_orig'], 
         lw=3, color='red', linestyle='--', label='TfidVectorizer Original')

plt.plot(eff_cached['fits'], eff_cached['cv_cached'], 
         lw=2.5, color='blue', linestyle='-', label='Count Vectorizer Memory Cached')
plt.plot(eff_cached['fits'], eff_cached['tv_cached'], 
         lw=3, color='blue', linestyle='--', label='TfidVectorizer Memory Cached')

plt.plot(eff_logan['fits'], eff_logan['cv_logan'], 
         lw=2.5, color='black', linestyle='-', label='Count Vectorizer New')
plt.plot(eff_logan['fits'], eff_logan['tv_logan'], 
         lw=3, color='black', linestyle='--', label='TfidVectorizer New')

plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Pipelines")
plt.legend(loc = 'best')
plt.show()
```


![png](PipelineEfficiency_files/PipelineEfficiency_63_0.png)


#### Visualization by Pipline


```python
eff_logan
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reps</th>
      <th>fits</th>
      <th>ss_logan</th>
      <th>pf_logan</th>
      <th>cv_logan</th>
      <th>tv_logan</th>
      <th>cv_td_logan</th>
      <th>tv_td_logan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>0.337648</td>
      <td>0.329182</td>
      <td>4.815905</td>
      <td>2.197995</td>
      <td>2.043521</td>
      <td>2.265842</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
      <td>0.336465</td>
      <td>0.344149</td>
      <td>6.568332</td>
      <td>2.512184</td>
      <td>2.158687</td>
      <td>2.040282</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>20</td>
      <td>0.341959</td>
      <td>0.336109</td>
      <td>8.806089</td>
      <td>2.928364</td>
      <td>2.333201</td>
      <td>2.061443</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>50</td>
      <td>0.437900</td>
      <td>0.423550</td>
      <td>16.460682</td>
      <td>4.176842</td>
      <td>2.383285</td>
      <td>2.382574</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>100</td>
      <td>0.434349</td>
      <td>0.542314</td>
      <td>28.065589</td>
      <td>6.179202</td>
      <td>2.338781</td>
      <td>2.343176</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75</td>
      <td>150</td>
      <td>0.562190</td>
      <td>0.641041</td>
      <td>40.617662</td>
      <td>8.128802</td>
      <td>2.553450</td>
      <td>2.371559</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>200</td>
      <td>0.619316</td>
      <td>0.646805</td>
      <td>50.271060</td>
      <td>10.255086</td>
      <td>2.554965</td>
      <td>2.452167</td>
    </tr>
    <tr>
      <th>7</th>
      <td>150</td>
      <td>300</td>
      <td>0.785087</td>
      <td>0.839261</td>
      <td>71.012454</td>
      <td>14.410503</td>
      <td>2.780835</td>
      <td>2.431730</td>
    </tr>
    <tr>
      <th>8</th>
      <td>200</td>
      <td>400</td>
      <td>0.887184</td>
      <td>0.883647</td>
      <td>90.469659</td>
      <td>18.092645</td>
      <td>2.831297</td>
      <td>2.582043</td>
    </tr>
    <tr>
      <th>9</th>
      <td>250</td>
      <td>500</td>
      <td>1.088669</td>
      <td>1.099337</td>
      <td>110.879169</td>
      <td>22.177632</td>
      <td>2.991412</td>
      <td>2.716854</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500</td>
      <td>1000</td>
      <td>1.838825</td>
      <td>1.630478</td>
      <td>226.336279</td>
      <td>42.359731</td>
      <td>4.005885</td>
      <td>3.290292</td>
    </tr>
    <tr>
      <th>11</th>
      <td>750</td>
      <td>1500</td>
      <td>2.096881</td>
      <td>2.278316</td>
      <td>334.308263</td>
      <td>62.534262</td>
      <td>4.399962</td>
      <td>3.818105</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1000</td>
      <td>2000</td>
      <td>2.996025</td>
      <td>2.956987</td>
      <td>453.728119</td>
      <td>82.371572</td>
      <td>5.327830</td>
      <td>4.707286</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2500</td>
      <td>5000</td>
      <td>5.997752</td>
      <td>6.383000</td>
      <td>1120.259022</td>
      <td>202.750489</td>
      <td>9.461915</td>
      <td>7.908639</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5000</td>
      <td>10000</td>
      <td>11.603223</td>
      <td>13.809455</td>
      <td>2160.935821</td>
      <td>406.772957</td>
      <td>16.150460</td>
      <td>13.800006</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure(figsize=(20,25))

ax1 = fig.add_subplot(3,2,1)
ax1.plot(eff_orig['fits'], eff_orig['ss_orig'], 
         lw=3, color='red', label='Original')
ax1.plot(eff_cached['fits'], eff_cached['ss_cached'], 
         lw=3, color='blue', label='Memory Cached')
ax1.plot(eff_logan['fits'], eff_logan['ss_logan'], 
         lw=3, color='black', label='New Method')
plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Standard Scalar Pipeline")
plt.legend(loc = 'best')
plt.grid()

ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(eff_orig['fits'], eff_orig['pf_orig'], 
         lw=3, color='red', label='Original')
ax2.plot(eff_cached['fits'], eff_cached['pf_cached'], 
         lw=3, color='blue', label='Memory Cached')
ax2.plot(eff_logan['fits'], eff_logan['pf_logan'], 
         lw=3, color='black', label='New Method')
plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Polynomial Feature Pipeline")
plt.legend(loc = 'best')
plt.grid()

ax3 = fig.add_subplot(3, 2, 3)
ax3.plot(eff_orig['fits'], eff_orig['cv_orig'], 
         lw=3, color='red', label='Original')
ax3.plot(eff_cached['fits'], eff_cached['cv_cached'], 
         lw=3, color='blue', label='Memory Cached')
ax3.plot(eff_logan['fits'], eff_logan['cv_logan'], 
         lw=3, color='black', label='New Method')
plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Count Vectorizer Pipeline")
plt.legend(loc = 'best')
plt.grid()

ax4 = fig.add_subplot(3, 2, 4)
ax4.plot(eff_orig['fits'], eff_orig['tv_orig'], 
         lw=3, color='red', label='Original')
ax4.plot(eff_cached['fits'], eff_cached['tv_cached'], 
         lw=3, color='blue', label='Memory Cached')
ax4.plot(eff_logan['fits'], eff_logan['tv_logan'], 
         lw=3, color='black', label='New Method')
plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit TfidVectorizer Pipeline")
plt.legend(loc = 'best')
plt.grid()

ax5 = fig.add_subplot(3, 2, 5)
ax5.plot(eff_orig['fits'], eff_orig['cv_td_orig'], 
         lw=3, color='red', label='Original')
ax5.plot(eff_cached['fits'], eff_cached['cv_td_cached'], 
         lw=3, color='blue', label='Memory Cached')
ax5.plot(eff_logan['fits'], eff_logan['cv_td_logan'], 
         lw=3, color='black', label='New Method')
plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to Fit Count Vectorizer with Truncation Pipeline")
plt.legend(loc = 'best')
plt.grid()

ax6 = fig.add_subplot(3, 2, 6)
ax6.plot(eff_orig['fits'], eff_orig['tv_td_orig'], 
         lw=3, color='red', label='Original')
ax6.plot(eff_cached['fits'], eff_cached['tv_td_cached'], 
         lw=3, color='blue', label='Memory Cached')
ax6.plot(eff_logan['fits'], eff_logan['tv_td_logan'], 
         lw=3, color='black', label='New Method')
plt.xlabel('# Fits')
plt.ylabel('Time (s)')
plt.title("Time to TfidVectorizor with Truncation Pipeline")
plt.legend(loc = 'best')
plt.grid()

plt.show()
```


![png](PipelineEfficiency_files/PipelineEfficiency_66_0.png)

