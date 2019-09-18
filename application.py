#%%
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
from collections import OrderedDict
from collections import namedtuple
DottedDict = namedtuple

## DECOMPOSITION
from sklearn.decomposition import NMF
from scipy.linalg import svd

### NLU
from ibm_watson import NaturalLanguageUnderstandingV1 as NaLaUn
from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions,ConceptsOptions,EntitiesOptions,KeywordsOptions,RelationsOptions,SyntaxOptions

### Presentation / apps
from matplotlib import pyplot as plt
import seaborn as sns

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_table.FormatTemplate as FormatTemplate
from dash_table.Format import Sign
from dash.dependencies import Input, Output

import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

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
                        categories= CategoriesOptions(),
                        concepts  = ConceptsOptions(),
                        entities  = EntitiesOptions(),
                        keywords  = KeywordsOptions(),
                        relations = RelationsOptions(),
                        syntax    = SyntaxOptions()
                        )

##### CLASS OBJECT FOR ARCHETYPAL ANALYSIS (UNDER CONSTRUCTION). ORIGINAL FUNCIONGIN BELOW CLASS OBJECT ########

class DocumentArchetypes:
    '''
    DocumentArchetypes performs Archetypal Analysis on a corpus consisting of a set of documents, for example a set 
    of articles, books, news stories or medical dictations.
    
    Input parameters:
    
    PATH            - Dictionary with paths to I/O
    PATH['data']    - Directory for input text files. Example: './data/input_texts/'
    PATH['results'] - Directory for output.           Example: './data/output_nlu/'
    
    NLU                   - Dictionary with information for running Watson NLU
    NLU['apikey']         - apikey for running Watson NLU
    NLU['apiurl']         - URL for Watson NLU API
    NLU['version']        - Watson NLU version, e.g. '2019-07-12'
    NLU['features']       - Features requested from Watson NLU for each document in the set, e.g. 
                                Features(
                                categories= CategoriesOptions(),
                                concepts  = ConceptsOptions(),
                                entities  = EntitiesOptions(),
                                keywords  = KeywordsOptions(),
                                relations = RelationsOptions(),
                                syntax    = SyntaxOptions()
                                )

    Attributes:

        
        self.PATH 
    
        
    '''
    from ibm_watson import NaturalLanguageUnderstandingV1 as NaLaUn
    from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions,ConceptsOptions,EntitiesOptions,KeywordsOptions,RelationsOptions,SyntaxOptions
    
    def __init__(self, PATH, NLU):
        self.PATH = PATH
        self.NLU  = NLU
        self.nlu_model  = NaLaUn(version=NLU['version'] , iam_apikey = NLU['apikey'], url = NLU['apiurl'])  #Local Natural Language Understanding object
        self.archetypes_dic = {}
        
        ################
        ## PREPARE DATA 
        ################
        self.filenames = os.listdir(self.PATH['data']) 
        self.dictation_dic = {}            #dictionary for dictation files
        for name in self.filenames:
            self.dictation_dic[name.replace('.txt','')] = open(self.PATH['data']+name).read()
        
        ###############################
        ## PERFORM WATSON NLU ANALYSIS
        ###############################
        
        self.watson = {}    #Dictionary with Watson-NLU results for each dictation
        
        self.watson_pkl = PATH['results']+'all_dictations_nlu.pkl'  
        pkl_exists = os.path.exists(self.watson_pkl)

        if pkl_exists:
            self.watson = pickle.load( open( self.watson_pkl, "rb" ) )

        else: #perform nlu-analysis on dictations
            for item in list(self.dictation_dic.items()):
                lbl  = item[0]
                text = item[1]
                self.watson[lbl] = self.nlu_model.analyze(text = text, features=NLU['features'])
                f = open(PATH['results']+str(lbl)+'_nlu.pkl','wb')
                pickle.dump(self.watson[lbl],f)
                f.close()

            f = open(self.watson_pkl,'wb')
            pickle.dump(self.watson,f)
            f.close() 

        # Copy Watson NLU results to Pandas Dataframes
        self.watson_nlu = {}
        for dctn in self.watson.items():
            self.watson_nlu[dctn[0]] = {}
            for item in list(dctn[1].result.items()):
                self.watson_nlu[dctn[0]][item[0]]=pd.DataFrame(list(item[1]))


    ##############
    # ARCHETYPAL ANALYSIS
    ##############

    def archetypes(self,typ='entities',n_archs=6,bootstrap = False, bootstrap_frac = 0.5):
        hyperparam = (n_archs,bootstrap,bootstrap_frac)
        if typ not in self.archetypes_dic.keys():
            self.archetypes_dic[typ] = {}
        if hyperparam not in self.archetypes_dic[typ].keys():
            self.archetypes_dic[typ][hyperparam] = {}
            df = pd.DataFrame()
            for key in self.watson_nlu:
                dfx = self.watson_nlu[key][typ].copy()
                dfx['dictation'] = key
                df = df.append(dfx,sort=True)
            if typ is 'entities':
                df = df[df['type']=='HealthCondition']
                df.rename({'relevance': 'rel0'}, axis=1,inplace=True)
                df['relevance'] = df['rel0'] * df['confidence']
            mat = df.pivot_table(index='dictation',columns='text',values='relevance').fillna(0)
            self.archetypes_dic[typ][hyperparam] = Archetypes(mat,n_archs,bootstrap = bootstrap, bootstrap_frac = bootstrap_frac)
        return self.archetypes_dic[typ][hyperparam]


    def display_archetype(self,typ = 'entities' , n_archs = 6, arch_nr = 0, var = 'variables', threshold = 0.1):
        if var is 'variables':
            arc = self.archetypes(typ,n_archs).f.T.sort_values(by=arch_nr,ascending = False)
            result = arc[
                        arc[arch_nr] >= (threshold * arc[arch_nr][0])
                     ]
            return result
        elif var is 'dictations':
            arc = sns.clustermap(archetypes(typ,n_archs).o).data2d
            return arc

                




#####  ORIGINAL DRAFT - PACKAGED AS A CLASS HERE ABOVE ##################### 

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


archetypes_dic = {}
def archetypes(typ='entities',n_archs=6):
    if typ not in archetypes_dic.keys():
        archetypes_dic[typ] = {}
    if n_archs not in archetypes_dic[typ].keys():
        archetypes_dic[typ][n_archs] = {}
        df = pd.DataFrame()
        for key in df_dic:
            dfx = df_dic[key][typ].copy()
            dfx['dictation'] = key
            df = df.append(dfx,sort=True)
        if typ is 'entities':
            df = df[df['type']=='HealthCondition']
            df.rename({'relevance': 'rel0'}, axis=1,inplace=True)
            df['relevance'] = df['rel0'] * df['confidence']
        mat = df.pivot_table(index='dictation',columns='text',values='relevance').fillna(0)
        archetypes_dic[typ][n_archs] = Archetypes(mat,n_archs)
    return archetypes_dic[typ][n_archs]



def display_archetype(typ = 'entities' , n_archs = 6, arch_nr = 0, var = 'variables', threshold = 0.1):
    if var is 'variables':
        arc = archetypes(typ,n_archs).f.T.sort_values(by=arch_nr,ascending = False)
        result = arc[
                    arc[arch_nr] >= (threshold * arc[arch_nr][0])
                 ]
        return result
    elif var is 'dictations':
        arc = sns.clustermap(archetypes(typ,n_archs).o).data2d
        return arc

    
  

#%%

## DASH/PLOTLY  WEB APP
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
application = app.server
app.title = 'IBM Watson – Natural Language Understanding'

app.layout = html.Div(
    html.Div([
        html.Div([
            html.H1(children='DICTATION-NLU',
                    className = "nine columns",
                    style={
                    'margin-top': 20,
                    'margin-right': 20
                    },
            ),
            
            dcc.Markdown(children='''
                        Archetypal Analysis of Medical Dictations. Process:
                        1. **Natural Language Understanding**:
                            - Dictations are analyzed by IBM Watson Natural Language Understanding. 
                            - Output variables: keywords, entities, concepts and categories.
                        2. **Archetypal Analysis**:
                            - Create Archetypes: Cluster data over variables, using NMF Non-zero Matrix Factorization
                            - *Try using TSNE*
                            - Dictations and variables are mapped onto the Archetypes/clusters
                        ''',
                    className = 'nine columns')
        ], className = "row"),
        html.Div([
            html.H2(children='ARCHETYPES:VARIABLES',
                    className = "nine columns",
                    style={
                    'margin-top': 20,
                    'margin-right': 20
                    },
            )
        ], className = "row"),
        html.Div(
                    [
                        html.Label('Variables', style={'font-weight' : 'bold'}),
                        dcc.Dropdown(
                            id = 'Var',
                            options=[
                                {'label': 'Keywords'  ,'value': 'keywords'},
                                {'label': 'Entities'  ,'value': 'entities'},
                                {'label': 'Concepts'  ,'value': 'concepts'},
                                {'label': 'Categories','value': 'categories'},
                            ],
                            value = 'keywords',
                        ) 
                    ],
                    className = 'two columns',
                    style={'margin-top': '30'}
                ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label('#Archetypes', style={'font-weight' : 'bold'}),
                        dcc.Dropdown(
                            id = 'NoA',
                            options=[{'label':k,'value':k} for k in range(2,100)],
                            value = 6,
                            multi = False
                        ) 
                    ],
                    className = 'one columns offset-by-one',
                    style={'margin-top': '30'}
                ),
                html.Div(
                    [
                        html.Label('Cut at', style={'font-weight' : 'bold'}),
                        dcc.Dropdown(
                            id = 'Threshold',
                            options=[{'label':str(k)+'%','value':k/100} for k in range(1,99)],
                            value = 0.1,
                            multi = False
                        ) 
                    ],
                    className = 'one columns offset-by-one',
                    style={'margin-top': '30'}
                ),
            ], className="row"
        ),
        html.Div([
            html.Div([
                dcc.Graph(
                    id='variables-heatmap'
                )
            ])
        ]),
        # html.Div([
        #     html.H3('DICTATIONS MAPPED ONTO ARCHETYPES'),
        #     dcc.Graph(id='dictations')
        # ])
     ])
)


@app.callback(
    dash.dependencies.Output('variables-heatmap', 'figure'),
    [dash.dependencies.Input('Var', 'value'),
     dash.dependencies.Input('NoA', 'value'),
     dash.dependencies.Input('Threshold', 'value')]
)

def arch_heatmap_variables(typ, n_archs, threshold):
    variables = (typ,n_archs,threshold)

    def f(i):
        return display_archetype(arch_nr=i,typ=typ,n_archs=n_archs,threshold=threshold).sort_values(by=i) #Sort by archetype i

    maxrows = int(1+ n_archs//3)
    cols = 3
    fig = make_subplots(rows=maxrows, cols=cols, horizontal_spacing=0.2)  
    for i in range(n_archs):
        fig.add_trace( go.Heatmap(  z = f(i),
                                    y = f(i).index,
                                    x = f(i).columns,
                                    xgap = 1,
                                    ygap = 1,
                        ), col = i%cols +1,row = int(i//cols)+1
            )
    fig.update_layout(height=400*maxrows, width=1200, title_text="Subplots")
    return fig



# @app.callback(
#     dash.dependencies.Output('dictations', 'figure'),
#     [dash.dependencies.Input('Var', 'value'),
#      dash.dependencies.Input('NoA', 'value'),
#      dash.dependencies.Input('Threshold', 'value')]
# )

# def arch_heatmap_dictations(typ, n_archs, threshold):
#     variables = (typ,n_archs,threshold)
#     f = archetypes(typ,n_archs).o
#     fig = go.Heatmap(   z = f
#             )
#     return fig



#%%

if __name__ == '__main__':
    app.run_server(port=8080, debug=True)


