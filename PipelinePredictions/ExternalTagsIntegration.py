import pandas as pd
import os
import sys
import json

tagpath = './PipelinePredictions/tag_data/'

# Dinamically add the library to sys.path
PATH = os.path.dirname(os.__file__)
sys.path.append(os.path.join(PATH, 'Libraries-GP'))

#eutop Libraries
from SQLServer import DATABASE_CONFIG_NEW, sql_cnnt

def getExternals():
    data = pd.read_sql_query("SELECT [id],[client_id],[vertical],[source],[acq_date]FROM [dbo].[co_verticals_ext] ",
                             sql_cnnt("cnnt", DATABASE_CONFIG_NEW))


    with open(tagpath + 'tags_verts_ext_mapping.json') as fp:
        tag_ext = json.load(fp)

    #'business to business (b2b)', 238: 'business to consumer (b2c)', 239: 'software', 240: 'hardware'
    matchhard = {'b2b':237, 'b2c':238, 'hardware':239, 'software':240}

    def processvert(x):
        arraio = x.split('/')
        out = []
        for jj in arraio:
            try:
                mss = tag_ext[jj]
                for i in mss['vertical']:
                    out.append(i)
            except:
                pass
        return out


    def processtag(x):
        arraio = x.split('/')
        out = []
        for aio in matchhard.keys():
            if aio in x:
                out.append(matchhard[aio])
        for jj in arraio:
            try:
                mss = tag_ext[jj]
                for i in mss['tag']:
                    out.append(i)
            except:
                pass
        return out

    data['vert'] = data.vertical.apply(lambda x : processvert(x))
    data['tag'] = data.vertical.apply(lambda x : processtag(x))

    def findnull(x):
        if len(x[0])>0 and len(x[1])>0:
            return 0
        else:
            return 1

    #return only clients mapped to something
    data['drop'] = data[["vert","tag"]].apply(lambda x: findnull(x), axis=1)
    data = data[data['drop'] == 0]
    data2 = data.groupby("client_id")["vert"].apply(sum).reset_index()
    data3 = data.groupby("client_id")["tag"].apply(sum).reset_index()

    data2['vert'] = data2['vert'].apply(set).apply(list)
    data3['tag'] = data3['tag'].apply(set).apply(list)

    data = data2.merge(data3, left_on='client_id', right_on='client_id')
    return data[['client_id', 'vert', 'tag']]

#from SQLServer import insert_into_sql_auto_incr
#insert_into_sql_auto_incr(conn_str = DATABASE_CONFIG_NEW, table_name = "tb_tags", db = b, ex_man = True)

