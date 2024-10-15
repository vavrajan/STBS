# Import global packages
import os
import time

import pandas as pd
from absl import app
from absl import flags

import numpy as np
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow_probability as tfp
import warnings
import matplotlib.pyplot as plt


us_states = {
    'AL': {'name': 'Alabama', 'region': 'Southeast'},
    'AK': {'name': 'Alaska', 'region': 'West'},
    'AZ': {'name': 'Arizona', 'region': 'West'},
    'AR': {'name': 'Arkansas', 'region': 'South'},
    'CA': {'name': 'California', 'region': 'West'},
    'CO': {'name': 'Colorado', 'region': 'West'},
    'CT': {'name': 'Connecticut', 'region': 'Northeast'},
    'DE': {'name': 'Delaware', 'region': 'Northeast'},
    'FL': {'name': 'Florida', 'region': 'Southeast'},
    'GA': {'name': 'Georgia', 'region': 'Southeast'},
    'HI': {'name': 'Hawaii', 'region': 'West'},
    'ID': {'name': 'Idaho', 'region': 'West'},
    'IL': {'name': 'Illinois', 'region': 'Midwest'},
    'IN': {'name': 'Indiana', 'region': 'Midwest'},
    'IA': {'name': 'Iowa', 'region': 'Midwest'},
    'KS': {'name': 'Kansas', 'region': 'Midwest'},
    'KY': {'name': 'Kentucky', 'region': 'Southeast'},
    'LA': {'name': 'Louisiana', 'region': 'South'},
    'ME': {'name': 'Maine', 'region': 'Northeast'},
    'MD': {'name': 'Maryland', 'region': 'Northeast'},
    'MA': {'name': 'Massachusetts', 'region': 'Northeast'},
    'MI': {'name': 'Michigan', 'region': 'Midwest'},
    'MN': {'name': 'Minnesota', 'region': 'Midwest'},
    'MS': {'name': 'Mississippi', 'region': 'South'},
    'MO': {'name': 'Missouri', 'region': 'Midwest'},
    'MT': {'name': 'Montana', 'region': 'West'},
    'NE': {'name': 'Nebraska', 'region': 'Midwest'},
    'NV': {'name': 'Nevada', 'region': 'West'},
    'NH': {'name': 'New Hampshire', 'region': 'Northeast'},
    'NJ': {'name': 'New Jersey', 'region': 'Northeast'},
    'NM': {'name': 'New Mexico', 'region': 'West'},
    'NY': {'name': 'New York', 'region': 'Northeast'},
    'NC': {'name': 'North Carolina', 'region': 'Southeast'},
    'ND': {'name': 'North Dakota', 'region': 'Midwest'},
    'OH': {'name': 'Ohio', 'region': 'Midwest'},
    'OK': {'name': 'Oklahoma', 'region': 'South'},
    'OR': {'name': 'Oregon', 'region': 'West'},
    'PA': {'name': 'Pennsylvania', 'region': 'Northeast'},
    'RI': {'name': 'Rhode Island', 'region': 'Northeast'},
    'SC': {'name': 'South Carolina', 'region': 'Southeast'},
    'SD': {'name': 'South Dakota', 'region': 'Midwest'},
    'TN': {'name': 'Tennessee', 'region': 'Southeast'},
    'TX': {'name': 'Texas', 'region': 'South'},
    'UT': {'name': 'Utah', 'region': 'West'},
    'VT': {'name': 'Vermont', 'region': 'Northeast'},
    'VA': {'name': 'Virginia', 'region': 'Southeast'},
    'WA': {'name': 'Washington', 'region': 'West'},
    'WV': {'name': 'West Virginia', 'region': 'Southeast'},
    'WI': {'name': 'Wisconsin', 'region': 'Midwest'},
    'WY': {'name': 'Wyoming', 'region': 'West'}
}
# created by Chat GPT...


def transform_name(name):
    commacount = name.count(', ')
    if commacount == 1:
        last_name, first_name = name.split(', ')
    elif commacount == 2:
        last_name, first_name, other = name.split(', ')
    else:
        last_name, first_name, other1, other2 = name.split(', ')
    #return f'{first_name.upper()} {last_name.upper()}'
    return f'{first_name.split()[0].upper()} {last_name.upper()}'

def get_surname(name):
    # first_name, last_name = name.split(' ')
    splitted_name = name.split(' ')
    last_name = splitted_name[-1].replace("'", "")
    return f'{last_name.split()[0].upper()}'

def get_n_surname(name):
    # Get initial from the first name and paste it with surname together
    splitted_name = name.split(' ')
    last_name = splitted_name[-1].replace("'", "")
    initial_name = splitted_name[0][0:1].upper()
    return f'{initial_name + "_" + last_name.split()[0].upper()}'

### Directories
data_name = 'hein-daily'
project_dir = os.getcwd()
source_dir = os.path.join(project_dir, "data", data_name)
orig_dir = os.path.join(source_dir, "orig")
data_dir = os.path.join(source_dir, "clean")

### Loading csv and creating a subset for session 114 of senate
congress = pd.read_csv(os.path.join(orig_dir, "data_aging_congress.csv"))
# congress['name'] = congress['bioname'].apply(transform_name)
# congress['surname'] = congress['name'].apply(get_surname)

# for i in range(97, 115):
for i in range(114, 115):
    congressi = congress[congress['congress'] == i]
    congressi['name'] = congressi['bioname'].apply(transform_name)
    congressi['surname'] = congressi['name'].apply(get_surname)

    # congressi.shape
    # congressi
    senatorsi = congressi[congressi['chamber'] == 'Senate']
    # senatorsi.shape


    ### Loading the author map used in TBIP to match senators
    addendum = str(i)
    #author_data = np.loadtxt(os.path.join(data_dir, "author_map" + addendum + ".txt"), delimiter=" ")
    author_info = pd.read_csv(os.path.join(data_dir, "author_info" + addendum + ".csv"))
    author_info["region"] = np.array([us_states[state]["region"] for state in author_info["state"]])


    ### Merging the datasets
    # author_info['name']
    # senatorsi['bioname']

    print("Define names, surnames, unique ids.")
    author_info['surname'] = author_info['name'].apply(get_surname)

    # Corrections of author_info to match senatorsi
    id = author_info.index[author_info.name == 'THAD COCHRAN'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'WILLIAM COCHRAN'

    if i == 99:
        senatorsi = pd.concat([senatorsi, congressi[congressi.surname == 'BROYHILL']])

    sur = 'MCCONNELL'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    id = author_info.index[author_info.name == 'MITCH MCCONNELL'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'Addison Mitchell (Mitch) MCCONNELL'

    sur = 'GRAMM'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    id = author_info.index[author_info.name == 'PHIL GRAMM'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'William Philip (Phil) GRAMM'

    sur = 'GRAHAM'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    id = author_info.index[author_info.name == 'BOB GRAHAM'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'Daniel Robert (Bob) GRAHAM'

    sur = 'SANFORD'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    id = author_info.index[author_info.name == 'JAMES SANFORD'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'James Terry SANFORD'
    id = senatorsi.index[senatorsi.bioname == 'SANFORD, (James) Terry'].tolist()
    if len(id) > 0:
        senatorsi['name'][id] = 'James Terry SANFORD'

    if i == 101:
        senatorsi = pd.concat([senatorsi, congressi[congressi.surname == 'AKAKA']])
        senatorsi = pd.concat([senatorsi, congressi[congressi.surname == 'COATS']])

    sur = 'LOTT'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    congressi["bioname"][congressi.surname == sur]
    id = author_info.index[author_info.name == 'TRENT LOTT'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'Chester Trent LOTT'

    sur = 'WYDEN'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    congressi["bioname"][congressi.surname == sur]
    id = author_info.index[author_info.name == 'RON WYDEN'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'Ronald Lee WYDEN'

    sur = 'ROBERTS'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    congressi["bioname"][congressi.surname == sur]
    id = author_info.index[author_info.name == 'PAT ROBERTS'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'Charles Patrick (Pat) ROBERTS'

    sur = 'ALLARD'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    congressi["bioname"][congressi.surname == sur]
    id = author_info.index[author_info.name == 'WAYNE ALLARD'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'A. Wayne ALLARD'

    sur = 'NELSON'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    congressi["bioname"][congressi.surname == sur]
    id = author_info.index[author_info.name == 'BEN NELSON'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'Earl Benjamin (Ben) NELSON'
    id = author_info.index[author_info.name == 'BILL NELSON'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'Clarence William (Bill) NELSON'

    if i == 109:
        senatorsi = pd.concat([senatorsi, congressi[congressi.surname == 'MENENDEZ']])

    sur = 'CORKER'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    congressi["bioname"][congressi.surname == sur]
    id = author_info.index[author_info.name == 'BOB CORKER'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'Robert (Bob) CORKER'

    if i == 110:
        senatorsi = pd.concat([senatorsi, congressi[congressi.surname == 'WICKER']])

    sur = 'HEITKAMP'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    congressi["bioname"][congressi.surname == sur]
    id = author_info.index[author_info.name == 'HEIDI HEITKAMP'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'Mary Kathryn (Heidi) HEITKAMP'

    sur = 'CRUZ'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    congressi["bioname"][congressi.surname == sur]
    id = author_info.index[author_info.name == 'TED CRUZ'].tolist()
    if len(id) > 0:
        author_info['name'][id] = 'Rafael Edward (Ted) CRUZ'


    ### Define first letter + surname as an identifier
    senatorsi['n_sur'] = senatorsi['name'].apply(get_n_surname)
    author_info['n_sur'] = author_info['name'].apply(get_n_surname)
    print("Names, surnames, unique ids defined.")

    # deal with duplicated keys:
    sur = 'BROWN'
    author_info[author_info.surname == sur]
    senatorsi["bioname"][senatorsi.surname == sur]
    congressi["bioname"][congressi.surname == sur]
    id = author_info.index[author_info.name == 'SCOTT BROWN'].tolist()
    if len(id) > 0:
        author_info['n_sur'][id] = 'SC_BROWN'
    id = author_info.index[author_info.name == 'SHERROD BROWN'].tolist()
    if len(id) > 0:
        author_info['n_sur'][id] = 'SH_BROWN'

    id = senatorsi.index[senatorsi.bioname == 'BROWN, Scott P.'].tolist()
    if len(id) > 0:
        senatorsi['n_sur'][id] = 'SC_BROWN'
    id = senatorsi.index[senatorsi.bioname == 'BROWN, Sherrod'].tolist()
    if len(id) > 0:
        senatorsi['n_sur'][id] = 'SH_BROWN'

    # senatorsi['name'].to_numpy()
    # author_info['name'].to_numpy()
    # senatorsi['surname'].to_numpy()
    # author_info['surname'].to_numpy()

    # matching on name --> some are not matched properly
    # merged = pd.merge(author_info, senatorsi, on='name', how='outer', indicator=True)
    # merged.shape
    # mismatched = merged[merged['_merge'] != 'both']
    # mismatched

    # matching on surname --> but what if some senators have the same surname?
    # merged = pd.merge(author_info, senatorsi, on=['surname'], how='left', indicator=True)
    # merged.shape
    # merged.columns
    # mismatched = merged[merged['_merge'] != 'both']
    # mismatched.name
    # merged[merged.surname_x.isin(["BYRD"])]
    # author_info[author_info.surname.isin(["BYRD"])]
    # senatorsi[senatorsi.surname.isin(["BYRD"])]

    # matching on first letter of a name combined with surname: n_sur
    # merged = pd.merge(author_info, senatorsi, on=['n_sur', 'surname'], how='left', indicator=True)
    merged = pd.merge(author_info, senatorsi, on=['n_sur'], how='left', indicator=True)
    # Still has some issues (somebody in author_info uses his/her second name instead of the first).
    # i=98, William Thad, COCHRANE goes by Thad COCHRANE in author_info not as William COCHRANE. --> creates NaNs
    # merged.shape
    # merged.columns
    print("author_info a senatorsi merged.")
    # Create some new columns:
    bins = pd.IntervalIndex.from_tuples([(0, 1), (1, 10), (10, 100)])
    merged['exper_cong'] = pd.cut(merged['cmltv_cong'], right=True,
                                  bins=bins)  # , labels=['Beginner', 'Advanced', 'Expert'])

    bins = pd.IntervalIndex.from_tuples([(0, 1), (1, 5), (5, 100)])
    merged['exper_chamber'] = pd.cut(merged['cmltv_chamber'], right=True,
                                     bins=bins)  # , labels=['Beginner', 'Advanced', 'Expert'])
    print("Exper_cong and exper_chamber defined.")

    ### Saving
    mergedsub = merged[['name_x', 'name_y', 'surname_x', 'surname_y', 'n_sur', 'gender', 'party', 'state', 'region',
                        'age_years', 'generation', 'cmltv_cong', 'cmltv_chamber', 'exper_cong', 'exper_chamber']]
    print("Subset of merged dataset is saved.")

    mergedsub.to_csv(os.path.join(data_dir, 'author_detailed_info' + addendum + '.csv'))

# ### Summary of the columns
# # Age
# merged['age_years'].min()
# merged['age_years'].mean()
# merged['age_years'].max()
# plt.hist(merged['age_years']); plt.show()
# plt.close()
#
# merged['generation'].value_counts()
#
# # Cumulative appearences in congress / senate
# merged['cmltv_cong'].value_counts()
# plt.hist(merged['cmltv_cong']); plt.show()
# plt.close()
# merged['cmltv_chamber'].value_counts()
# plt.hist(merged['cmltv_chamber']); plt.show()
# plt.close()
#
# # Experience
# bins = pd.IntervalIndex.from_tuples([(0, 1), (1, 10), (10, 100)])
# merged['exper_cong'] = pd.cut(merged['cmltv_cong'], right=True, bins=bins) #, labels=['Beginner', 'Advanced', 'Expert'])
# merged['exper_cong'].value_counts()
#
# bins = pd.IntervalIndex.from_tuples([(0, 1), (1, 5), (5, 100)])
# merged['exper_chamber'] = pd.cut(merged['cmltv_chamber'], right=True, bins=bins) #, labels=['Beginner', 'Advanced', 'Expert'])
# merged['exper_chamber'].value_counts()
#
# # Political party
# merged['party'].value_counts()
# merged['party_code'].value_counts()
#
# # State
# merged['state'].value_counts()
# merged['state_abbrev'].value_counts()
# merged['region'].value_counts()
#

### Updating session 114
i = 114
addendum = str(i)
np.sum(mergedsub['surname_x'] != mergedsub['surname_y'])
mergedsub['surname'] = mergedsub['surname_x']
mergedsub = mergedsub[['name_x', 'name_y', 'surname', 'gender', 'party', 'state', 'region',
                       'age_years', 'generation', 'cmltv_cong', 'cmltv_chamber', 'exper_cong', 'exper_chamber']]

mergedsub.to_csv(os.path.join(data_dir, 'author_detailed_info' + addendum + '.csv'))


## Adding religion
rel_info = pd.read_csv(os.path.join(orig_dir, "data_religion_114.csv"))
rel_info['surname'] = rel_info['name_2'].apply(get_surname)
rel_info['surname'][46] = 'MANCHIN'
rel_info['surname']


mergedsub['surname'].to_numpy()
rel_info['surname'].to_numpy()
np.setdiff1d(rel_info['surname'].to_numpy(), mergedsub['surname'].to_numpy())
np.setdiff1d(mergedsub['surname'].to_numpy(), rel_info['surname'].to_numpy())
rel_merged = pd.merge(mergedsub, rel_info, on='surname', how='inner', indicator=True)
# rel_merged = pd.merge(rel_info, mergedsub, on='surname', how='inner', indicator=True)
rel_merged.shape
rel_merged.columns
rel_merged['surname'].to_numpy()
rel_merged['RELIGION'].value_counts()
rel_merged['party'].value_counts()
rel_merged['PARTY'].value_counts()
rel_merged['cmltv_cong'].value_counts()
rel_merged['exper_cong'].value_counts()
rel_merged['cmltv_chamber'].value_counts()
rel_merged['exper_chamber'].value_counts()
rel_merged['FRESHMAN.'].value_counts()
rel_merged['CHAMBER'].value_counts()
rel_merged['generation'].value_counts()

rel_mergedsub = rel_merged[['name_x', 'name_y', 'surname', 'gender', 'party', 'state', 'region',
                            'age_years', 'generation', 'cmltv_cong', 'cmltv_chamber', 'exper_cong', 'exper_chamber',
                            'RELIGION', 'FRESHMAN.']]

rel_mergedsub.to_csv(os.path.join(data_dir, 'author_detailed_info_with_religion' + addendum + '.csv'))