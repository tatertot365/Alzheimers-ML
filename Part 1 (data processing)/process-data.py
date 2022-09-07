# Import necessary libraries
import pandas as pd
from chembl_webresource_client.new_client import new_client

# Target search for coronavirus
target = new_client.target
target_query = target.search('acetylcholinesterase')
targets = pd.DataFrame.from_dict(target_query)

selected_target = targets.target_chembl_id[0]

activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

df = pd.DataFrame.from_dict(res)

df.to_csv('acetylcholinesterase_01_bioactivity_data_raw.csv', index=False)

df2 = df[df.standard_value.notna()]
df2 = df2[df.canonical_smiles.notna()]

len(df2.canonical_smiles.unique())
df2_nr = df2.drop_duplicates(['canonical_smiles'])

selection = ['molecule_chembl_id','canonical_smiles','standard_value']
df3 = df2_nr[selection]

df3.to_csv('acetylcholinesterase_02_bioactivity_data_preprocessed.csv', index=False)
df4 = pd.read_csv('acetylcholinesterase_02_bioactivity_data_preprocessed.csv')

bioactivity_threshold = []
for i in df4.standard_value:
  if float(i) >= 10000:
    bioactivity_threshold.append("inactive")
  elif float(i) <= 1000:
    bioactivity_threshold.append("active")
  else:
    bioactivity_threshold.append("intermediate")

bioactivity_class = pd.Series(bioactivity_threshold, name='class')
df5 = pd.concat([df4, bioactivity_class], axis=1)

df5.to_csv('acetylcholinesterase_03_bioactivity_data_curated.csv', index=False)

