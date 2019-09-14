#########Script to add new PID and Root Cause to True prediction = 0#########
import pandas as pd
import pymongo
from Utils import *
settings = configparser.ConfigParser()
settings.read('config.ini')

key = "csap_stage_database"
db = get_db(settings, key)


pf = "ASR9000" #"CRS"
rc_coll = 'IQS_FARC_Labels_'collection = db[rc_coll + str(pf)]
cursor = collection.find({})
new_rc_df = pd.DataFrame(list(cursor))


d_1 = {}
ass_fa = new_rc_df['FA_CASE_NUMBER'].tolist()
rc = new_rc_df['ROOT_CAUSE'].tolist()
for i in range(len(ass_fa)):
    d_1[ass_fa[i]] = []


#For each FA, we are storing it's Root cause in a dictionary
#This dictionary we later use to add rows in the dataframe as we iterate through the FA cases for each SR.
for i in range(len(ass_fa)):
    d_1[ass_fa[i]].append(rc[i])


#Appending RC to a list
rec_df = pd.read_csv() #Give the path to the excel sheet
newdf = rec_df[rec_df['True_prediction'] == 1] #This dataframe has to be only true prediction == 0
ass_fa = newdf['Associated FA'].tolist() 
rc_list = []
for i in range(len(ass_fa)):
    k = []
    l = list(set(ass_fa[i].strip(',').split(',')))
    for j in l:
        k.append(d[str(j)])
    rc_list.append(list(set(k[0])))


pids = []
for i in range(len(ass_fa)):
    k = []
    l = list(set(ass_fa[i].strip(',').split(',')))
    for j in l:
        a = d_1[str(j)]
        #a = a.split(',')
        a = [item for item in a if (not item[-1:].isdigit())]
        k.append(a)
    pids.append(list(set(k[0])))
