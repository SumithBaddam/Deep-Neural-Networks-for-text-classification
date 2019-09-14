######Accuracy computation using FA data######
import pandas as pd
import pymongo
import sys
import shutil
import argparse
sys.path.insert(0, "/data/ingestion/")
from Utils import *

#Parser options
options = None
def parse_options():
    parser = argparse.ArgumentParser(description="""Script to classify SR cases""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    args = parser.parse_args()   
    return args

options = parse_options()
settings = configparser.ConfigParser()
settings.read('config.ini')

if(options.env.lower() == "prod"):
    key = "csap_prod_database"

elif(options.env.lower() == "stage"):
    key = "csap_stage_database"

db = get_db(settings, key)

rec_df = pd.read_excel('IQSTextAnalytics_10.xlsx', encoding='iso-8859-1')
rec_df.dropna(subset=['Recommended FA'], inplace=True)
rec_df.drop_duplicates(subset ="Associated FA", inplace = True)

#df = pd.read_csv('IQS_FARC_Data_Extracted.csv', encoding='utf-8') # CRS
#df = pd.read_csv('IQS_FARC_Extracted_ASR9000.csv', encoding='utf-8') #ASR9000

#pf = 'ASR9000'
coll2 = 'IQS_SRNotes_Extracted_ASR9000_Orig'
collection2 = db[coll2] #+ str(pf)]
cursor = collection2.find({})
df = pd.DataFrame(list(cursor))

rec_df['Associated FA'] = rec_df['Associated FA'].fillna('')

ass_fa = rec_df['Associated FA'].tolist()
for i in range(len(ass_fa)):
    ass_fa[i] = ass_fa[i].replace(' ', '').split(',')

rec_fa = rec_df['Recommended FA'].tolist()
for i in range(len(rec_fa)):
    rec_fa[i] = rec_fa[i].strip("'[]").replace(" '", '').replace("'", '').split(',')

d = {}
for i in range(df.shape[0]):
    fa = list(set(df.iloc[i]['ASSOSCIATED_FA'].strip(',').split(',')))
    for j in fa:
        d[j] = [] #df.iloc[i]['ITEM_NUMBER']

for i in range(df.shape[0]):
    status = False
    fa_list = list(set(df.iloc[i]['ASSOSCIATED_FA'].strip(',').split(',')))
    for fa in fa_list:
        if(df.iloc[i]['ASSOSCIATED_CPN'].strip(',') != ''):
            status = True
            #fa = list(set(df.iloc[i]['ASSOSCIATED_FA'].strip(',').split(',')))[0]
            for j in df.iloc[i]['ASSOSCIATED_CPN'].strip(',').split(','):
                d[fa].append(j[:-3])
        if(df.iloc[i]['ASSOSCIATED_CDETS'].strip(',') != ''):
            status = True
            #fa = list(set(df.iloc[i]['ASSOSCIATED_FA'].strip(',').split(',')))[0]
            for j in df.iloc[i]['ASSOSCIATED_CDETS'].strip(',').split(','):
                d[fa].append(j)
        if(status == False):
            for j in df.iloc[i]['ASSOSCIATED_PID'].strip(',').split(','):
                d[fa].append(j)

    #else:
        #d[df.iloc[i]['FA_CASE_NUMBER']].append(df.iloc[i]['ITEM_NUMBER'])
    #d[df.iloc[i]['FA_CASE_NUMBER']].append(df.iloc[i]['ITEM_NUMBER'][:-3])

c = 0
true_preds = [0]*len(ass_fa)
good_cases_1 = []
status = False
for i in range(len(ass_fa)):
    print(ass_fa[i])
    for j in ass_fa[i]:
        for k in rec_fa[i]:
            for l in d[j]:
               if(l in d[k]):
                   c = c + 1
                   true_preds[i] = 1
                   good_cases_1.append([j, k])
                   status = True
                   break
            if(status == True):
                #status = False
                break
        if(status == True):
            status = False
            break

print('Avoidable FA:', round((float(c/rec_df.shape[0]))*100, 2), '%')
rec_df['True_prediction'] = true_preds
newdf = rec_df[rec_df['True_prediction'] == 1]
falsedf = rec_df[rec_df['True_prediction'] == 0]

ass_fa = newdf['Associated FA'].tolist()
for i in range(len(ass_fa)):
    ass_fa[i] = ass_fa[i].replace(' ', '').split(',')

rec_fa = newdf['Recommended FA'].tolist()
for i in range(len(rec_fa)):
    rec_fa[i] = rec_fa[i].strip("'[]").replace(" '", '').replace("'", '').split(',')


status = False
for i in range(len(ass_fa)):
    for j in ass_fa[i]:
        status = [0]*len(rec_fa[i])
        for m in range(len(rec_fa[i])):
            k = rec_fa[i][m]
            #print(k, i, m)
            for l in d[j]:
               if(l in d[k]):
                   status[m] = 1
                   #print(k, 'true')
                   break
    for n in range(len(status)):
        if(status[n] == 0):
            rec_fa[i][n] = 0

for i in range(len(rec_fa)):
    rec_fa[i] = list(filter(lambda a: a != 0, rec_fa[i]))

newdf['Recommended FA'] = rec_fa

######Adding Root Cause data######
#pf = "CRS" #"ASR9000" 
pf = "ASR9000" 
rc_coll = 'IQS_FARC_Labels_'
'''
username = "csaprw"
passwd = "csaprw123"
hostname = "sjc-wwpl-fas4"
port = "27017"
db = "csap_prd"
mongo_connection_string="mongodb://"+username+":"+passwd+"@"+hostname+":"+port+"/"+db
client=pymongo.MongoClient(mongo_connection_string)
db=client.get_database(db)
'''
collection = db[rc_coll + str(pf)]
cursor = collection.find({})
rc_df = pd.DataFrame(list(cursor))
new_rc_df = rc_df #rc_df[rc_df['ITEM_TYPE']=='PID']

#Storing RC in a dict
d = {}
ass_fa = new_rc_df['FA_CASE_NUMBER'].tolist()
rc = new_rc_df['ROOT_CAUSE'].tolist()
for i in range(len(ass_fa)):
    d[ass_fa[i]] = []

for i in range(len(ass_fa)):
    d[ass_fa[i]].append(rc[i])

#Appending RC to a list
ass_fa = newdf['Associated FA'].tolist()
rc_list = []
for i in range(len(ass_fa)):
    k = []
    l = list(set(ass_fa[i].split(', ')))
    for j in l:
        k.append(d[str(j)])
    rc_list.append(list(set(k[0])))

pids = []
for i in range(len(ass_fa)):
    k = []
    l = list(set(ass_fa[i].split(', ')))
    for j in l:
        a = d[str(j)]
        #a = a.split(',')
        a = [item for item in a if (not item[-2:].isdigit())]
        k.append(a)
    pids.append(list(set(k[0])))

newdf['Root_Cause'] = rc_list
newdf['PID'] = pids
newdf.drop(['Cluster Id'], axis=1)
newdf['Similar_SR'] = "'" + newdf['Similar_SR'] + "'"

finaldf = newdf.append(falsedf)
finaldf.to_csv('IQSTextAnalytics_v2_'+pf+'_new.csv', encoding = 'utf-8', index = False)
