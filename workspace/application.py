#++++++++++++++++++++++++++++++++++++++++++++++
# Before running the script, edit 
# 'SET HYPERPARAMETERS' 
# - the rest is automated
#++++++++++++++++++++++++++++++++++++++++++++++

##################
### IMPORT MODULES
##################

### System
import sys
import os

### I/O
import json
import pickle

### General Processing
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import math
import collections

## DECOMPOSITION
from sklearn.decomposition import NMF
from scipy.linalg import svd

### NLU
from ibm_watson import NaturalLanguageUnderstandingV1 as NaLaUn
from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions,ConceptsOptions,EntitiesOptions,KeywordsOptions,RelationsOptions,SyntaxOptions

### Presentation
from matplotlib import pyplot as plt
import seaborn as sns

## GENERAL FUNCTIONS 
### NORMALIZATION
#### Statistic normalization - subtract mean, scale by standard deviation
def norm_stat(vec, weights = False):
    '''
    Normalizes a vector v-v.mean())/v.std() 
    '''
    if weights:
        return  np.mean(abs(vec - vec.mean()))  
    return (vec-vec.mean())/vec.std()

#### Algebraic normalization - dot product
def norm_dot(vec, weights = False):
    '''
    Normalizes a vector - dot product: v @ v = 1
    '''
    if weights:
        return  np.sqrt(vec @ vec)
    
    return vec / np.sqrt(vec @ vec)

#### Algebraic normalization - dot product
def norm_sum(vec, weights = False):
    '''
    Normalizes a vector - sum: v.sum = 1
    '''
    if weights:
        return  vec.sum()
    
    return vec / vec.sum()

#### Scaled Normalization -
def scale(vec, weights = False):
    '''
    Normalizes a vector: v.min = 0, v.max = 1
    '''
    stop_divide_by_zero = 0.00000001
    if weights:
        return (vec.max()-vec.min() + stop_divide_by_zero)
    return (vec-vec.min())/(vec.max()-vec.min() + stop_divide_by_zero)
def cleanup_chars(string,char_list = ('\n',' ')):
    result = string
    for char in char_list:
        result = result.replace(char,'')
    return result

##########################################
### IBM-WATSON/NLU API-KEY (DON'T EDIT)
##########################################
# The script asks for the API key when run. 
# Do NOT save API-Keys in the code. 

local_dir_exists = os.path.exists('.local')
if not local_dir_exists:
    os.mkdir('.local')
    
credentials_exists = os.path.exists('.local/crd.env')
if not credentials_exists:
    print('Credentials needed for https://cloud.ibm.com/catalog/services/natural-language-understanding )')
    apikey = input(prompt='API-Key?')
    apiurl = input(prompt='API-URL?')
    crd = open('.local/crd.env','w')
    crd.write(  'NATURAL_LANGUAGE_UNDERSTANDING_IAM_APIKEY='+apikey)
    crd.write('\nNATURAL_LANGUAGE_UNDERSTANDING_URL='       +apiurl)  
    

# dian_pkl_file = PATH['results']+'all_dictations_nlu.pkl'  
# dian_pkl_exists = os.path.exists(dian_pkl_file)
# if 'apikey' not in locals():
#     apikey = input(prompt='API-Key? ( https://cloud.ibm.com/catalog/services/natural-language-understanding )')  


# # MATRIX-FACTORIZATION: DIMENSIONALITY REDUCTION & ARCHETYPING

# ## CLUSTER FEATURES INTO OCCUPATION CATEGORIES
# ## Use non-zero matrix factorization for clustering
# ## Use singular value decomposition first state for determining overall similarity


class Archetypes:
    '''
    Archetypes: Performs NMF of order n on X and stores the result as attributes. 
    Archetypes are normalized: cosine similarity a(i) @ a(i) = 1. 
    Atributes:
        my_archetypes.n         - order / number of archetypes
        my_archetypes.X         - input matrix
        
        my_archetypes.model     - NMF model 
        my_archetypes.w         - NMF w-matrix 
        my_archetypes.h         - NMF h-matrix
        
        my_archetypes.o         - occupations x archetypes matrix (from w-matrix)
        my_archetypes.on        - occupations x normalized archetypes matrix (from w-matrix) - SOCP number as index. 
        my_archetypes.occ       - occupations x normalized archetypes matrix - Occupation names as index
        
        my_archetypes.f         - features x archetypes matrix (from h-matrix)
        my_archetypes.fn        - features x normalized archetypes matrix
        
    '''
    def __init__(self,X,n,norm = norm_dot):
        self.n = n
        self.X = X
        self.model = NMF(n_components=n, init='random', random_state=0, max_iter = 1000, tol = 0.0000001)
        self.w = self.model.fit_transform(self.X)
        self.o = pd.DataFrame(self.w,index=self.X.index)
        self.on = self.o.T.apply(norm).T
        self.occ = self.on.copy()
        self.occ['Occupations'] = self.occ.index
#        self.occ['Occupations'] = self.occ['Occupations'].apply(onet_socp_name)
        self.occ = self.occ.set_index('Occupations')
        self.h = self.model.components_
        self.f = pd.DataFrame(self.h,columns=X.columns)
        self.fn =self.f.T.apply(norm).T
        self.plot_occupations_dic ={}
        self.plot_features_dic ={}

        
    def plot_features(self,fig_scale = (1,3.5),metric='cosine', method = 'single',vertical = False): 
        '''
        Plot Archetypes as x and features as y. 
        Utilizes Seaborn Clustermap, with hierarchical clustering along both axes. 
        This clusters features and archetypes in a way that visualizes similarities and diffferences
        between the archetypes. 
        
        Archetypes are normalized (cosine-similarity): dot product archetype[i] @ archetype[i] = 1.
        The plot shows intensities (= squared feature coefficients) so that the sum of intensities = 1.  

        fig_scale: default values (x/1, y/3.5) scales the axes so that all feature labels are included in the plot.
        
        For other hyperparameters, see seaborn.clustermap
     
        '''
        param = (fig_scale,metric,method,vertical)
        if param in self.plot_features_dic.keys():
            fig = self.plot_features_dic[param]
            return fig.fig

        df = np.square(self.fn)

        if vertical:
            fig = sns.clustermap(df.T,robust = True, z_score=1,figsize=(
                self.n/fig_scale[0],self.X.shape[1]/fig_scale[1]),method = method,metric = metric)        
        else: # horizontal
            fig = sns.clustermap(df,robust = True, z_score=0,figsize=(
                self.X.shape[1]/fig_scale[1],self.n/fig_scale[0]),method = method,metric = metric)        
        self.features_plot = fig
        return fig


    def plot_occupations(self,fig_scale = (1,3.5),metric='cosine', method = 'single',vertical = False):
        '''
        Plot Archetypes as x and occupations as y. 
        Utilizes Seaborn Clustermap, with hierarchical clustering along both axes. 
        This clusters occupations and archetypes in a way that visualizes similarities and diffferences
        between the archetypes. 
        
        Occupations are normalized (cosine-similarity): dot product occupation[i] @ occupation[i] = 1.
        The plot shows intensities (= squared feature coefficients) so that the sum of intensities = 1.  

        fig_scale: default values (x/1, y/3.5) scales the axes so that all feature labels are included in the plot.
        
        For other hyperparameters, see seaborn.clustermap
     
        '''
        param = (fig_scale,metric,method,vertical)
        if param in self.plot_occupations_dic.keys():
            fig = self.plot_occupations_dic[param]
            #return
            return fig.fig

        df = np.square(self.occ)
        if vertical:
            fig = sns.clustermap(df, figsize=(
                self.n/fig_scale[0],self.X.shape[0]/fig_scale[1]),method = method,metric = metric)
        else: # horizontal
            fig = sns.clustermap(df.T, figsize=(
                self.X.shape[0]/fig_scale[1],self.n/fig_scale[0]),method = method,metric = metric)
        self.plot_occupations_dic[param] = fig
        #return
        return fig.fig


class Svd:
    ''''
    Singular value decomposition-as-an-object
        my_svd = Svd(X) returns
        my_svd.u/.s/.vt – U S and VT from the Singular Value Decomposition (see manual)
        my_svd.f        – Pandas.DataFrame: f=original features x svd_features
        my_svd.o        - Pandas.DataFrame: o=occupations x svd_features
        my_svd.volume(keep_volume) 
                        - collections.namedtuple ('dotted dicionary'): 
                          Dimensionality reduction. keeps 'keep_volume' of total variance
                          
                          
    '''
    def __init__(self,X):
        self.u,self.s,self.vt = svd(np.array(X))
        self.f = pd.DataFrame(self.vt,columns=X.columns)
        self.o = pd.DataFrame(self.u,columns=X.index)
        
    def volume(self,keep_volume):
        ''' 
        Dimensionality reduction, keeps 'keep_volume' proportion of original variance
        Type: collections.namedtuple ('dotted dictionary')
        Examples of usage:
        my_svd.volume(0.9).s - np.array: eigenvalues for 90% variance 
        my_svd.volume(0.8).f - dataframe: features for 80% variance
        my_svd.volume(0.5).o - dataframe: occupations for 50% variance      
        '''
        dotted_dic = collections.namedtuple('dotted_dic', 's f o')
        a1 = self.s.cumsum()
        a2 = a1/a1[-1]
        n_max = np.argmin(np.square(a2 - keep_volume))
        cut_dic = dotted_dic(s= self.s[:n_max],f= self.f.iloc[:n_max], o= self.o.iloc[:n_max])
        return cut_dic
        


##########################
## SET HYPERPARAMATERS
#### edit below ##########

# Import credentials
cred = open('.local/crd.env','r').read()
apikey,apiurl = cred.replace('NATURAL_LANGUAGE_UNDERSTANDING_IAM_APIKEY=','').replace(
                            'NATURAL_LANGUAGE_UNDERSTANDING_URL=','').split()

PATH = {}
PATH['data']    = '../data/Documents/'
PATH['results'] = './Watson-nlu-results/'

NLU = {}
NLU['apikey']         = apikey
NLU['apiurl']         = apiurl
NLU['version']        = '2019-07-12'
NLU['features']       = Features(
                        categories= CategoriesOptions(limit=4),
                        concepts  = ConceptsOptions(limit=20),
                        entities  = EntitiesOptions(limit=20),
                        keywords  = KeywordsOptions(limit=20),
                        relations = RelationsOptions(),
                        syntax    = SyntaxOptions()
                        )

nlu = NaLaUn(version=NLU['version'] , iam_apikey = NLU['apikey'], url = NLU['apiurl'])  #Local Natural Language Understanding object

################
## PREPARE DATA 
################
filenames = os.listdir(PATH['data']) 
dictation_dic = {}            #dictionary for dictation files
for name in filenames:
    dictation_dic[name.replace('.txt','')] = open(PATH['data']+name).read()

    
# Treat dictations_dic as 
# - dict when type(key)=str, eg dictation_dic['12'] -> value for key '12'
# - list when type(key)=int, eg dictation_dic[12] -> 12th value in dictionary 
def select_dictation(key):
    if type(key) is int:
        aa = list(dictation_dic.values())[key]
    else:
        aa = dictation_dic[key]
    return aa
dn = select_dictation           # dn <- Alias for select_dictation

###############################
## PERFORM WATSON NLU ANALYSIS
###############################
dictation_analysis = {}
dian = dictation_analysis

# If dictation_analysis dictionary already exists - read the pickled file
# If it does NOT already exist, perform calculations. 
dian_pkl_file = PATH['results']+'all_dictations_nlu.pkl'  
dian_pkl_exists = os.path.exists(dian_pkl_file)

if dian_pkl_exists:
    dian = pickle.load( open( dian_pkl_file, "rb" ) )

else: #perform nlu-analysis on dictations
    for item in list(dictation_dic.items()):
        lbl  = item[0]
        text = item[1]
        dian[lbl] = nlu.analyze(text = text, features=NLU['features'])
        f = open(PATH['results']+str(lbl)+'_nlu.pkl','wb')
        pickle.dump(dian[lbl],f)
        f.close()

    f = open(dian_pkl_file,'wb')
    pickle.dump(dian,f)
    f.close()  

# Transform dian to Pandas Dataframes
df_dic = {}
for dctn in dian.items():
    df_dic[dctn[0]] = {}
    for item in list(dctn[1].result.items()):
        df_dic[dctn[0]][item[0]]=pd.DataFrame(list(item[1]))

##############
# ARCHETYPAL ANALYSIS
##############

df = pd.DataFrame()

for key in df_dic:
    dfx = df_dic[key]['concepts'].copy()
    dfx['dictation'] = key
    df = df.append(dfx,sort=True)

mat = df.pivot('dictation','text','relevance')
m = mat.fillna(0)

archetypes = {}

n = 10 # Select number of Archetypes
mar = Archetypes(m,n)
archetypes[n] = {}
for i in range(n):
    archetypes[n][i] = mar.f.iloc[i].sort_values(ascending=False)[:20]
    print(str(i+1)+' of '+str(n))
    print(archetypes[n][i])
    print('\n')

