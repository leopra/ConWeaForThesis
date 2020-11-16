import pandas as pd
import os
import sys
from PipelinePredictions import Pipeline_predict as pipred
# Dinamically add the library to sys.path
PATH = os.path.dirname(os.__file__)
sys.path.append(os.path.join(PATH, 'Libraries-GP'))

#eutop Libraries
from SQLServer import DATABASE_CONFIG_NEW, sql_cnnt

data = pd.read_sql_query("select distinct client_id from co_aggregations where client_id not in "
                         "(select client_id from co_verticals)",
                         sql_cnnt("cnnt", DATABASE_CONFIG_NEW))


qry_desc = """select client_id, [description], source, acq_date from
(select client_id, [description], source, acq_date,row_number()
over(partition by client_id, source order by acq_date desc) as rn
from co_descriptions) as T
where rn = 1"""

qry_pitch = """select client_id, pitch_line, source, acq_date from 
(select client_id, pitch_line, source, acq_date,row_number() 
over(partition by client_id, source order by acq_date desc) as rn
from co_pitch_lines) as T
where rn = 1"""

priority_pitch = ['cbinsights', 'owler','i3', 'dealroom','crunchbase','old_db','linkedin']

priority_desc = ['cbinsights', 'crunchbase', 'dealroom', 'linkedin', 'old_db', 'wb_text_nlp', 'wb_html_position']


desc = pd.read_sql_query(qry_desc, sql_cnnt("cnnt", DATABASE_CONFIG_NEW))
pitch = pd.read_sql_query(qry_pitch, sql_cnnt("cnnt", DATABASE_CONFIG_NEW))

for index in data.values:
    desc_client = desc[desc.client_id == index]
    desc_client["source"] = pd.Categorical(desc_client["source"], categories=priority_desc, ordered=True)
    desc_client = desc_client.sort_values('source')

    pitch_client = pitch[pitch.client_id == index]
    pitch_client["source"] = pd.Categorical(pitch_client["source"], categories=priority_pitch, ordered=True)
    pitch_client = pitch_client.sort_values('source')

    preds_desc = pipred.generate_pseudo_labels(desc_client.description.values)
    preds_pitch = pipred.generate_pseudo_labels(desc_client.description.values)
    print(preds_desc)
    print(preds_pitch)