# python ./fa_sr_analysis.py --env prod
#########Modeling the SR data#############
import pandas as pd
import re
import nltk
from nltk.corpus import brown
import sys
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from gensim import corpora
from gensim.models.ldamodel import LdaModel as Lda
import pickle
import _pickle as cPickle
import json
import pymongo
import os
import numpy as np
import configparser
import sys
import shutil
import argparse
sys.path.insert(0, "/data/ingestion/")
from Utils import *
#from nltk.tag.stanford import NERTagger
from nltk.tag.stanford import StanfordNERTagger
from collections import Counter
import enchant
from nameparser.parser import HumanName

#d = enchant.Dict("en_US")
st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz', 'stanford-ner.jar')

#Parser options
options = None
def parse_options():
    parser = argparse.ArgumentParser(description="""Script to classify SR cases""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    args = parser.parse_args()   
    return args

#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')
collection_prefix = 'SRNotes_' #filepath = str(settings.get("SR_Source","srFilesLoc"))
#model_path = '/data/csap_models/srData/' #'/auto/vgapps-cstg02-vapps/analytics/csap/models/files/sr/'

#####Database configuration#####
options = parse_options()
if(options.env.lower() == "prod"):
    key = "csap_prod_database"
elif(options.env.lower() == "stage"):
    key = "csap_stage_database"

db = get_db(settings, key)
print(db)

######NLP parameters config######
brown_train = brown.tagged_sents(categories='news')
regexp_tagger = nltk.RegexpTagger(
    [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
     (r'(-|:|;)$', ':'),
     (r'\'*$', 'MD'),
     (r'(The|the|A|a|An|an)$', 'AT'),
     (r'.*able$', 'JJ'),
     (r'^[A-Z].*$', 'NNP'),
     (r'.*ness$', 'NN'),
     (r'.*ly$', 'RB'),
     (r'.*s$', 'NNS'),
     (r'.*ing$', 'VBG'),
     (r'.*n t$', 'VBG'),
     (r'.*ed$', 'VBD'),
     (r'.*', 'NN')
])

unigram_tagger = nltk.UnigramTagger(brown_train, backoff=regexp_tagger)
bigram_tagger = nltk.BigramTagger(brown_train, backoff=unigram_tagger)

cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
cfg["JJ+JJ"] = "JJ"
cfg["JJ+NN"] = "NNI"


class NPExtractor(object):
    def __init__(self, sentence):
        self.sentence = sentence
    # Split the sentence into singlw words/tokens
    def tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens
    # Normalize brown corpus' tags ("NN", "NN-PL", "NNS" > "NN")
    def normalize_tags(self, tagged):
        n_tagged = []
        for t in tagged:
            if t[1] == "NP-TL" or t[1] == "NP":
                n_tagged.append((t[0], "NNP"))
                continue
            if t[1].endswith("-TL"):
                n_tagged.append((t[0], t[1][:-3]))
                continue
            if t[1].endswith("S"):
                n_tagged.append((t[0], t[1][:-1]))
                continue
            n_tagged.append((t[0], t[1]))
        return n_tagged
    # Extract the main topics from the sentence
    def extract(self):
        tokens = self.tokenize_sentence(self.sentence)
        tags = self.normalize_tags(bigram_tagger.tag(tokens))
        status = True
        #print(tags)
        matches2 = []
        for t in tags:
            if t[1] == "NNP" or t[1] == "NNI":
            #if t[1] == "NNP" or t[1] == "NNI" or t[1] == "NN":
                matches2.append(t[0])       
        if(len(matches2) < 2):
            status = False
        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = "%s+%s" % (t1[1], t2[1])
                value = cfg.get(key, '')
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = "%s %s" % (t1[0], t2[0])
                    pos = value
                    tags.insert(x, (match, pos))
                    break
        matches = []
        #print(tags)
        #print(status)
        if(status == True):
            for t in tags:
                if t[1] == "NNP" or t[1] == "NNI":
                    matches.append(t[0])
        else:
            for t in tags:
                if t[1] == "NNP" or t[1] == "NNI" or t[1] == "NN":
                    matches.append(t[0])
        return matches


######Preprocessing of data######
stop = set(stopwords.words('english'))
#stoplist = set('done ensure work also detail case reply sr function https browser windows mozilla chrome safari link lead user use type important to from update log all apac active contact status attachment index web html free regardto cisco http mycase subject sincerely ist january february haven possible aim tmr helpdesk fine save load support note description done response timely query request hours yet regard service notification device kindly hi hey good email manager employee id test option time please details file attachment thanks regards state program ok look no yes type insert asr903 detailasr903 can thank mac macintosh when #name dear that and other www html name com en be have ll will here use make people know many call include part find become like | mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask would way something need things want every str please details look note refer to detail with the to is a as am I have been this for . ( )'.split())
stoplist = set('SR show skill workgroup think answers expire title solution timestamp descr communication ganesh input output est keep template iox decision replicate duplicate new known adam max may cust customer large small per answer amazon template delivered note notes closed creation admin install reliance workshop workaround region jio satellite zero impact business log additional logs unknown level product process step ctc collect unicom rma  utc % object unknown na xr ios iosxr xe shipment cc gmt attach attachments note attachments cisco operation webex page exception available not check legal illegal reddy mgmt arvind amer search srch sumeet thu addr focus afterhours sev pri manually automatically manual automatic node provide pid subtechnology access line name history brief johnson store management source tool william confirm confirmation error errors khtml click account download trap busy version option options bug debug share upload owner info chicago california sub initial role tech final permission unexpected expect misqueue requeue queue job sood overall ext exception share luther side jim didn internet poland katarzyna such doc michael confirmation key value docs php public contributor contribute source fileid content japan china desk service complete data fuller phythian adi ushaditya profile mazur comment home alert verify private greet sort open technical local director manager engineer location example deliver kara hannam adi type export definitions america americas helloadi span case cases high low create plan action current ship div group reason tac adi style font color colour instructions shipent dispatch last first contract require order pst ist gbt false true close sale sales category hay mon tue wed thurs fri size htts system syatems priority customer app application google engineer phone text epic problem company receive  person zip city  address  hi hello thanks regards wish cancel attach attachment kind summary thank description please end severity issue https http rtp for help other safe code county country need standard serial cisco.com  cisco com dot  india us  refer to and in with same number request \n \t \b request  also use make people know many call include part find become like mean often different usually take wikt come give well get since type list say sr also asa can thank #name dear email day when session por port txt that and other www html name com en be have ll will here use make people know many call include part find become like | mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask also use make people know many call include part find become like mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask would way something need things want every str please crs asr asa done cisco use asa make com sr team html cause web save subject sure thanks regards ensure work also detail case reply sr function https browser windows mozilla chrome safari link lead user use type important to from update log all apac active contact status attachment index web html free regardto cisco http mycase subject sincerely ist january february haven possible aim tmr helpdesk fine save load support note description done response timely query request hours yet regard service notification device kindly hi hey good email manager employee id test option time please details file attachment thanks regards state program ok look no yes type insert asr903 detailasr903 can thank mac macintosh when #name dear that and other www html name com en be have ll will here use make people know many call include part find become like | mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask would way something need things want every str please details look note refer to detail with the to is a as am I have been this for . ( )'.lower().split())
stoplist = {'Yinyan', 'satellite', 'zero', 'ctc', 'process', 'product', 'step', 'business', 'unknown', 'log', 'logs', 'additional', 'impact', 'level', 'collect', 'unicom', 'rma', '%', 'zr', 'ios', 'ioszr', 'utc', 'object', 'unkown', 'well', 'address', 'notification', 'misqueue', 'fri', 'permission', 'log', 'team', 'details', 'us', 'manager', 'february', 'session', 'helloadi', 'engineer', 'asa', 'attachments', 'dispatch', 'cisco.com', 'good', 'group', 'regard', 'webex', 'part', 'Street', 'apac', 'way', 'call', 'done', 'want', 'come', 'com', 'state', 'have', 'search', 'am', 'ensure', 'Beijing', 'font', 'iii', 'technical', 'status', 'side', 'trap', 'str', 'HW', 'kind', 'doc', 'update', 'export', 'gmt', 'reason', 'such', 'hi', 'fine', 'been', 'all', 'email', 'key', 'application', 'country', 'contact', 'home', 'debug', 'port', 'manual', 'user', 'cases', 'instructions', 'job', 'customer', 'since', 'response', 'important', 'jim', 'ship', 'need', 'include', 'automatically', 'sure', 'free', 'subject', 'wikt', 'to', 'en', 'check', 'and', 'fuller', 'ext', 'create', 'mycase', 'require', 'phythian', 'reddy', 'cancel', 'windows', 'mac', 'txt', 'Wu', 'public', 'give', 'mean', 'subtechnology', 'afterhours', 'cc', 'unexpected', 'change', 'Newark', 'verify', 'file', 'exception', 'open', 'other', 'person', 'manually', 'helpdesk', 'america', 'management', 'ok', 'service', 'http', 'asr903', 'sr', 'please', 'contributor', 'help', 'boston', 'fileid', 'app', 'deliver', 'desk', 'Stevens', 'program', 'find', 'poland', 'ask', 'that', 'automatic', 'know', 'order', 'be', 'crs', 'srch', 'hours', 'operation', 'index', 'Marco', 'things', 'Halsey', 'click', 'tac', 'available', 'tool', 'tech', 'time', 'web', 'mozilla', 'tmr', 'google', 'overall', 'info', 'type', 'johnson', 'queue', 'version', 'Hu', 'brief', 'mgmt', 'end', 'Microloops', 'thurs', 'Tang', 'contribute', 'function', 'TX', 'focus', 'standard', 'php', 'illegal', 'detailasr903', 'work', 'attach', 'this', 'haven', 'list', 'code', 'style', 'addr', 'safe', 'description', 'store', 'confirm', 'thu', 'reply', 'number', 'id', 'shipment', 'dear', 'dot', 'contract', 'luther', 'also', 'ist', 'company', 'Point', 'local', 'rtp', 'often', 'epic', 'look', 'name', 'wed', 'county', '#name', 'profile', 'the', 'page', 'i', 'timely', 'sub', 'kinds', 'lead', 'true', 'with', 'size', 'will', 'january', 'sales', 'value', 'safari', 'every', 'https', 'test', '.', 'final', 'problem', 'regards', 'something', 'ushaditya', 'note', 'data', 'role', 'yes', 'india', '(', 'serial', 'people', ')', 'sood', 'text', 'browser', 'make', 'amer', 'like', 'pid', 'options', 'internet', 'mon', 'katarzyna', 'thanks', 'sev', 'private', 'owner', 'sincerely', 'html', 'thank', 'different', 'khtml', 'for', 'city', 'wish', 'BRINGDOWN', 'New', 'div', 'color', 'history', 'adi', 'michael', 'll', 'possible', 'as', 'upload', 'active', 'content', 'hey', 'greet', 'query', 'errors', 'node', 'phone', 'account', 'aisne', 'Thoms', 'kara', 'say', 'location', 'plan', 'no', '\x08', 'false', 'error', 'line', 'pas', 'yet', 'Hamburg', 'didn', 'use', 'in', 'close', 'tue', '|', 'sumeet', 'span', 'comment', 'would', 'busy', 'cisco', 'por', 'shipent', 'share', 'china', 'usually', 'take', 'docs', 'macintosh', 'same', 'action', 'a', 'from', 'see', 'last', 'system', 'insert', 'many', 'Wisconsin', 'request', 'arvind', 'summary', 'zip', 'expect', 'Japan', 'definitions', 'high', 'here', 'Europe', 'charleston', 'hello', 'complete', 'example', 'link', 'japan', 'pst', 'not', 'source', 'hay', 'hannam', 'confirmation', 'colour', 'employee', 'low', 'kindly', 'sandy', 'NSHUT', 'william', 'download', 'case', 'priority', 'become', 'sale', 'actually', 'requeue', 'Kai', 'device', 'mazur', 'issue', 'current', 'california', 'regardto', 'asr', 'support', 'provide', 'attachment', 'get', 'chrome', 'refer', 'severity', 'save', 'legal', 'load', 'first', 'access', 'initial', 'when', 'chicago', 'detail', 'bug', 'is', 'sort', 'aim', 'day', 'alert', 'www', 'List', 'americas', 'category', 'pri', 'receive', 'gbt', 'can', 'htts', 'option', 'director', 'systems'}
stoplist = {'show', 'skill', 'workgroup', 'think', 'answers', 'ganesh', 'max', 'new', 'cust', 'template', 'iox', 'replicate', 'input', 'communication', 'solution', 'title', 'timestamp', 'descr', 'output', 'keep', 'known', 'max', 'may', 'customer', 'duplicate', 'replicate', 'known', 'est', 'stanley', 'expire', 'large', 'small', 'per', 'answer', 'amazon', 'template', 'delivered', 'note', 'notes', 'closed', 'creation', 'admin', 'install', 'region', 'workshop', 'workaround', 'zero', 'reliance', 'jio', 'impact', 'adam', 'php', 'focus', 'unknown', 'to', 'download', 'i', 'thanks', 'Heraghty', 'trap', 'style', 'Toth', 'Ashwin', 'Erero', 'attachment', 'ushaditya', '\x08', 'query', 'poland', 'note', 'E.', 'BUSADRERR', 'Mazur', 'Aguilar', 'hay', 'operation', 'usually', 'Dimitri', 'Beijing', 'thurs', 'way', 'Feng', 'webex', 'lead', 'Kara', 'Faisal', 'collect', 'www', 'Ravi', 'Sydney', 'want', 'Tamaela', 'issue', 'thank', 'Arthur', 'utc', 'rtp', 'say', 'US', 'htts', 'Pacific', 'Balakrishnan', 'sunil', 'refer', 'know', 'Rodrigo', 'sood', 'description', 'Harsh', 'por', 'end', 'impact', 'Loganatha', 'http', 'port', 'low', 'arvind', 'Katarzyna', 'Singh', 'action', 'cisco', 'States', 'high', 'font', 'technical', 'close', 'manual', 'John', 'Khairi', 'Durelle', 'web', 'Poovalingam', 'Raziel', 'company', 'Rick', 'USA', 'Zhang', 'LA', 'Cho', 'no', 'change', 'Cao', 'sincerely', 'str', 'Kazuma', 'Xiangbo', 'Hu', 'Mohd', 'same', 'as', 'Kadubeesanahalli', 'automatic', 'Shang', 'internet', 'detailasr903', 'Alam', 'important', 'available', 'since', 'true', 'standard', 'Daniel', 'customer', 'actually', 'system', 'asr', 'Cherian', 'Duke', 'Tencent', 'Kroehle', 'contributor', 'can', 'mgmt', 'Kyle', 'NC', 'engineer', 'Christina', 'Cisco', 'open', 'Takahashi', 'Enviado', 'Simmons', 'store', 'elmhurst', 'color', 'Espero', 'program', 'dispatch', 'Crashinfo', 'role', 'Street', 'Tang', 'Sergio', 'Almeida', 'country', 'desk', 'attach', 'Martha', 'need', 'is', 'Fei', 'busy', 'Chakravarti', 'unexpected', 'day', 'have', 'product', 'HW', 'XingYi', 'give', 'william', 'Dino', 'cases', 'Hola', 'Ishak', 'FW', 'huang', 'work', 'ok', 'click', 'ioszr', 'find', 'Fuller', 'support', 'SAN', 'details', 'Dear', 'ctc', 'Lalit', 'timely', 'upload', 'Japan', 'txt', 'Cynthia', 'be', 'Wisconsin', 'definitions', 'Coria', 'profile', '.', 'South', 'Nan', 'Abdulhadi', 'Ki', 'Tian', 'type', 'Hong', 'home', 'account', 'Fumitoshi', 'priority', 'debug', 'wish', 'load', 'Somuri', 'group', 'category', 'Mr.', 'Jalet', 'india', 'Sanaullah', 'last', 'contact', 'Angel', 'Sarjapur', 'comment', 'for', 'with', 'help', 'side', 'asa', 'log', 'phythian', 'location', 'first', 'North', 'ios', 'Phone', 'Hou', 'that', 'often', 'helpdesk', 'Yan', 'confirm', 'tac', 'Peng', 'Ganesh', 'Bhadauria', 'Johnson-Rustvold', 'brief', 'america', 'Arturo', 'Li', 'local', 'johnson', 'Dharmarajan', 'Ligang', 'Aleksander', 'version', 'come', 'G.', 'key', 'director', 'Marco', 'become', 'page', 'Haifeng', 'id', 'Shuang', 'legal', 'also', 'hello', 'mac', 'MR', 'Morgan', 'Guillermo', 'equipo', 'index', 'Kumar', 'ist', 'yes', 'tech', 'michael', 'provide', 'Cary', 'Mingyao', 'aim', 'Dheeraj', 'shipent', 'Cassidy', 'El', 'Yamamoto', 'and', 'subject', 'queue', 'Gabriel', 'Ce', 'Ping', 'https', 'India', 'alert', 'Proto', 'shipment', 'Polisetty', 'cisco.com', 'Hoerauf', 'Amanda', 'final', 'a', 'Joao', 'detail', 'Moonyong', 'kinds', 'active', 'Bill', 'kindly', 'adi', 'job', '&', 'Huevertech', 'Muthu', 'weerasekara', 'RTP', 'request', 'LC', 'Brian', 'instructions', 'sure', 'HongGil', 'ask', 'pri', 'Australia', 'Liu', 'pid', 'sandy', 'Zeledon', 'PoE', 'name', 'Xiaodong', 'Muthusamy', 'use', 'service', 'kind', 'level', 'David', 'Arvind', 'Anran', 'sales', 'ext', '\\', 'Miguel', 'Mohan', 'options', 'James', 'windows', 'Shenzhen', 'sr', 'BANGALORE', 'Jim', 'when', 'Condict', 'sumeet', 'complete', 'sort', 'melendez', 'california', 'private', 'errors', 'take', 'this', 'code', 'arias', 'session', 'order', 'macintosh', 'China', 'the', 'Miller', 'process', 'access', 'from', 'Huaibin', 'americas', 'line', 'expect', 'Maaz', 'please', 'Hi', 'S.', 'systems', 'Sarath', 'Macau', 'search', 'hi', 'Pack', 'option', 'google', 'response', 'wed', 'fine', 'requeue', 'mycase', 'Hannam', 'people', 'employee', ')', 'colour', 'Pei', 'Poland', 'Africa', 'regard', 'Dolores', 'Kong', 'J.', 'Mr', 'hannam', 'Yin', 'jim', 'confirmation', 'done', 'app', 'sev', 'Pls', 'create', 'asr903', 'fileid', 'Gabriela', 'deliver', 'county', 'many', 'logs', 'problem', 'Babu', 'chicago', 'Eva', 'PARK', 'Guangzhou', 'Longfei', 'notification', 'fri', 'Santhana', 'Mikhailov', 'dot', 'pas', 'verify', 'file', 'boston', 'case', 'something', 'Hofmann', 'Junichi', 'exception', 'Hays', 'iii', 'owner', 'Ishwar', 'well', 'text', 'Kai', 'zip', 'management', 'Requeue', 'Carlos', 'Wei', 'apac', 'Dhan', 'safe', 'browser', 'Umesh', 'katarzyna', 'thu', 'would', 'greet', 'kara', 'receive', 'Burns', 'Wenlong', 'source', 'Durrani', 'Hello', 'tool', 'include', 'am', 'error', 'sub', 'Chen', 'Huawen', 'New', 'Kathleen', 'Lawrenceville', 'cc', 'manually', 'html', 'div', 'Matecki', 'DeCooman', 'Yang', 'zr', 'Sridhar', 'afterhours', 'Jeffrey', 'example', 'different', 'sale', 'span', 'gbt', 'Bob', 'satellite', 'good', 'user', 'Type', 'automatically', 'Rivalino', 'Marathalli', 'zero', 'time', 'Garcia', 'll', 'docs', 'TRIANGLE', 'subtechnology', 'Varthur', 'Peggy', 'chrome', 'Morales', 'Gonzalez', 'current', 'team', 'such', 'aisne', 'manager', 'Pandav', 'fuller', 'haven', 'value', 'Hanashiro', 'insert', 'Andrew', 'illegal', 'Patel', 'Xu', 'List', 'Maazkhan', 'Suresh', 'angel', 'Pandian', 'Jiang', 'Hobli', 'Pearse', 'Europe', 'Subramanian', 'Leo', 'Thoms', 'Daniels', 'number', 'Gerardo', 'Tim', 'serial', 'Alfaro', 'pst', 'reason', 'like', 'Slidel', 'Carpenter', 'doc', 'regards', 'KHTML', 'free', 'bug', 'hours', 'Aravind', 'Grettel', 'ship', 'Krishna', 'addr', 'Bai', 'possible', 'Saludos', 'Zhao', 'Asia', 'business', 'plan', 'Gheevarghese', 'reply', 'person', 'Hirokazu', 'Microloops', 'Harris', 'rma', 'unkown', 'Guerra', 'node', 'Ma', 'Yi', 'Emma', 'Weerasekara', 'Prinkesh', 'size', 'hey', 'Masaki', 'Dion', 'Hua', 'Manila', '%', 'january', 'Victor', 'export', 'Eric', 'Please', 'Sunil', 'wikt', 'TX', 'device', 'Brown', 'Masahiro', 'Xing', 'here', 'Max', 'attachments', 'safari', 'test', 'William', 'every', 'miguel', 'Pd', 'Raghu', 'Boyd', 'link', 'Iwamochi', 'unicom', 'part', 'Adi', 'crs', 'in', 'permission', 'Sendhil', 'city', 'status', 'share', 'NSHUT', 'ALAMOSA', 'charleston', 'Rodriguez', 'Rawat', 'LEI', 'Newark', 'didn', 'tmr', 'state', 'khtml', '|', 'tue', 'Vanessa', 'Mansoor', 'function', 'M.S', 'Ettwejiri', 'America', 'Sharma', 'email', 'mazur', 'Oman', 'Jason', 'object', 'Arias', 'Libya', 'ORR', 'Seoul', 'require', 'mon', 'get', 'severity', 'other', 'info', 'mean', 'Christopher', 'contribute', 'Slidell', 'all', 'misqueue', 'epic', 'look', 'update', 'Thatipalli', 'Amazon', 'California', 'content', 'Prakash', 'himanshu', 'Barragan', 'yet', 'Team', 'Antoniades', 'Halsey', '#name', 'amer', 'false', 'mozilla', 'overall', 'been', 'Dorenkamp', 'address', 'dear', 'make', 'japan', 'contract', 'data', 'Michael', 'Stevens', 'save', 'additional', 'Wang', 'BRINGDOWN', 'Wu', 'Mora', 'helloadi', 'summary', 'Mon-Thu', 'Vasu', 'Minghai', 'com', 'step', 'Ningbo', 'Choi', 'Wanxin', 'reddy', 'Chico', 'cancel', 'luther', 'gmt', 'Yao', 'Miki', 'en', 'see', 'regardto', 'not', 'Yinyan', 'Sumeet', 'Yue', 'Sood', 'Alser', 'will', 'history', 'february', 'Point', 'china', 'Nguyen', 'Korea', 'Barreto', 'us', 'Kim', 'Hokeun', 'Hamburg', 'Gibson', 'call', 'list', 'public', 'phone', 'Hill', 'initial', '(', 'ensure', 'check', 'things', 'Blackburn', 'Melendez', 'michigan', 'United', 'srch', 'application'}
stoplist = {'churn', 'feb', 'only', 'david', 'dumps', 'dump', 'purpose', 'purposes', 'pkt', 'label', 'ieee', 'start', 'bad', 'initiate', 'recent', 'jul', 'event', 'budafoki', 'but', 'up.', 'init', 'paul', 'project', 'projects', 'threshold', 'engineer', 'engineering', 'æœºç®±å‡çº§standby', 'xinyuan', 'municipal', 'financial', 'kompetencia', 'jtæ–°æ±äº¬', 'sep', 'onsite', 'kiss', 'into', 'kÃ¡lmÃ¡n', 'Ã¡rpÃ¡d', 'ã‚·ãƒªã‚¢ãƒ«ç•ªå·', 'august', 'date', 'confirmed', 'warning', 'directory', 'set', 'sho', 'dir', 'instal', 'main', 'none', 'count', 'now', 'although', 'clear', 'sesiÃ³n', 'atenciÃ³n', 'set', 'ingenieros', 'cuando', 'sesiones', 'msgq', 'faxç•ªå·', 'non', 'iosxr', 'july', 'alarm', 'cold', 'waiting', 'usage', 'ordered', 'filter', 'down', 'richmond', 'pass', 'try', 'don', 'digital', 'full', 'joÃ£o', 'mike', 'dallas', 'louis', 'mobile', 'laptop', 'device', 'kishiki', 'saitama', 'currently', 'available', 'profile', 'default', 'status', 'dev', 'prd', 'kill', 'killing', 'exit', 'enter', 'previous', 'nodeid', 'id', 'jrdhqiaf', 'eta', 'ordernumber', 'present', 'report', 'reports', 'asking', 'ask', 'new', 'newly', 'old', 'louis', 'beijing', 'agreed', 'ready', 'rochester', 'sat', 'sun', 'aug', 'mon', 'tue', 'web', 'jan', 'mar', 'april', 'june', 'jun', 'douglas', 'single', 'multiple', 'tired', 'condition', 'instance', 'host', 'console', 'closing', 'debug', 'hello', 'macintosh', 'Jeffrey', 'Rumana', 'time', 'private', 'need', 'ios', 'Grettel', 'Site', 'ushaditya', 'requeue', 'André', 'Jake', 'Englewood', 'Aurora', 'Danilo', 'Simmons', 'Phone', 'details', 'Hays', 'code', 'action', 'arias', 'as', 'Dhan', 'automatically', 'Tim', 'Sood', 'verify', 'Rommon', 'Logan', 'Mike', 'Xiaodong', 'sr', 'Doug', 'brief', 'amazon', 'can', 'WARSAW', 'JAPAN', 'Coutinho', 'Srikant', 'Kevin', 'Pls', 'Ferenc', 'Kathleen', 'Fan', 'Oklahoma', 'sandy', 'contributor', 'Thoms', 'jim', 'windows', 'Makoto', 'Ricardo', 'Jalet', 'country', 'USA', 'Toth', 'please', 'people', 'to', 'initial', 'Chen', 'Htom', 'manual', 'Rua', 'Andre', 'Yamamoto', 'asr903', 'Harada', 'that', 'angel', 'workshop', 'Aravind', 'List', 'Yohei', 'Leonel', 'Hong', 'Seoul', 'operation', 'Thiago', 'Alexey', 'download', 'SAN', 'look', 'Juan', 'Port', 'Figueira', 'Sharma', 'poland', 'johnson', 'Waldemar', 'Hannam', 'Halsey', 'Vanessa', 'sure', 'Wien', 'en', 'Chris', 'Haneef', 'Canada', 'crs', 'click', 'Raghu', 'Manila', 'Jim', 'management', 'Merritt', 'contact', 'help', 'Varthur', 'Stefan', 'Jiang', 'Pacific', 'person', 'Tshoot', 'know', 'Webb', 'Dolores', 'Hou', 'Harsh', 'be', 'Silva', 'satellite', 'Papua', 'Sousa', 'Erwin', 'Shane', 'am', 'Loopback', 'Kadubeesanahalli', 'example', 'Cengiz', 'Andrew', 'sales', 'error', 'Type', 'Slidell', 'logs', 'Rodrigues', 'katarzyna', 'Prakash', 'employee', 'miguel', 'Ahmad', 'Pd', ';', 'Michigan', 'Ln', 'Shafi', 'Amanda', 'Pei', 'Guillermo', 'every', 'India', 'Ko', 'Heffernan', 'Lineside', 'Zhang', 'Hobli', 'address', 'alert', 'Risky', 'Joao', 'Goutham', 'trap', 'Patel', 'query', 'll', 'use', 'Hua', 'give', 'Carpenter', 'SE', 'safe', 'Slidel', 'Roseville', 'yet', 'Sunil', 'keep', 'for', 'Guangzhou', 'Charles', 'BUSADRERR', 'sood', 'Wang', 'SAITAMA', 'name', 'Ishwar', 'call', 'Vyatskaya', 'timestamp', 'Shelfmgr', 'SHENZHEN', 'Yi', 'with', 'phythian', 'Harddisk', 'KHTML', 'reddy', 'Samir', 'Mode', 'Hola', 'iox', 'Ochi', 'ST', 'Galiev', 'https', 'Luis', 'Dechant', 'Alfaro', 'queue', 'plan', 'when', 'Pandav', 'Hirokazu', 'reply', 'php', 'Tian', 'Didier', 'Bill', 'deliver', 'Fuller', '\x08', 'account', 'Rendon', 'Minghai', 'Janos', 'Alser', 'mac', 'Requeue', 'county', 'find', 'Haider', 'Kazuma', 'Jay', 'Morgan', 'Almirante', '.', 'impact', 'Ronnie', 'side', 'list', 'Moscow', 'Wanxin', 'shipment', 'errors', 'day', 'Amol', 'Angel', 'Cole', 'thanks', 'standard', 'equipo', 'charleston', 'BANGALORE', 'Sendhil', 'California', 'Ningbo', 'Nome', 'BUDAPEST', 'Linz', 'LIANG', 'Morobe', 'Douglas', 'last', 'would', 'Aliya', 'Acosta', 'AZ', 'Yue', 'Muthusamy', 'Vonda', 'Yan', 'Zheng', 'Kohlmann', 'H', 'Eva', 'change', 'zr', 'object', 'Firmensitz', 'Nord', 'Masaki', 'Russ', 'Cho', 'access', 'Marathalli', 'Dian', 'sale', 'Krishna', 'Mark', 'Xu', 'Lisowski', 'Berlin', 'Garren', 'Matthew', 'RTP', 'Luiz', 'Saitama', 'answer', 'way', 'require', 'FW', 'a', 'google', 'mazur', 'new', 'Lawrenceville', 'L', 'Kris', 'wish', 'Dores', 'NVRAM', 'Mohan', 'Kumar', 'Erero', 'Stemmons', 'ext', 'João', 'description', 'Tencent', 'Elodie', 'timely', 'Ritzl', 'Street', 'Daniels', 'Michael', 'check', 'Pedro', 'and', 'Shubham', 'Pete', 'systems', 'profile', 'mon', 'Haluza', 'local', 'James', 'greet', 'Santhana', 'ISP', 'Frank', 'Email', 'receive', 'think', 'Thatipalli', 'Toronto', 'work', 'Adi', 'thank', 'Darrell', 'Jacomini', 'Aguilar', 'tue', 'Safi', 'El', 'helpdesk', 'sev', 'rtp', 'browser', 'contribute', 'NC', 'come', 'Brian', 'ok', 'Hellos', 'MR.', 'STOCKTON', 'function', 'www', 'creation', 'Walnut', 'Paulo', 'install', 'different', 'chrome', 'california', 'Morales', 'workgroup', 'Duke', 'Nguyen', 'View', 'cust', 'something', 'cancel', 'PoE', 'Costa', 'Zok', 'hannam', 'Hsieh', 'Mohd', 'Nan', 'regard', 'insert', 'html', 'node', 'Gabriel', 'unkown', 'Pearl', 'application', 'Lu', 'Mililani', 'reliance', 'confirmation', 'illegal', 'Dorenkamp', 'fileid', 'Charlotte', 'Longfei', 'Idogawa', 'Bauer', 'Norriton', 'Xiangbo', 'free', 'Gilks', 'william', 'Sartori', 'us', 'Babu', 'Aguero', 'Dave', 'Dion', 'Sydney', 'P', 'subtechnology', 'Boyd', 'Kalman', 'michael', 'log', 'UFA', 'LC', 'Gerardo', 'Dresch', 'Lambrechts', 'span', 'Mr', 'february', 'Budapest', 'Hugo', 'store', 'luther', 'Europe', 'Masahiro', 'China', 'Bokaro', 'Richard', 'Endo', 'Ford', 'Carlos', 'Khairi', 'ist', 'good', 'definitions', '&', 'Mariaux', 'Omiya', 'Newark', 'content', 'VSC', 'HongGil', 'RSP0', 'Greenville', 'wikt', 'Tony', 'Three', 'america', 'Vitaly', 'Paul', 'Watts', 'kind', 'pid', 'Mountain', 'severity', 'Sie', 'Akriti', 'ganesh', 'M.S', 'subject', 'Shenzhen', 'japan', 'Vestal', 'ALAMOSA', 'Rodriguez', 'other', 'Ce', 'doc', 'Balakrishnan', 'DUBAI', 'Field', 'Milovan', 'ENOC', 'sort', 'been', 'York', 'Liu', 'Rafael', 'BOUCA', 'Wu', 'usually', 'team', 'Shota', 'Point', 'important', 'output', 'say', 'Lake', 'Jid', 'Kendal', 'Haug', 'Xie', 'active', 'include', 'South', 'Salamanderstraße', 'Haifeng', 'Mingyao', 'font', 'Castro', 'Aschenbrenner', 'Olsen', 'Miss', 'Pincode', 'colour', 'may', 'stanley', 'Luciano', 'amer', 'phone', 'search', 'update', 'Mais', 'Parc', 'same', 'Cao', 'weerasekara', 'wed', 'serial', 'dear', 'Salt', 'Rich', 'Hsueh', 'option', 'Arthur', 'Abdulhadi', 'cc', 'Leal', 'response', 'Sridhar', 'gmt', 'mgmt', 'Garcia', 'Kong', 'notification', 'Str', 'COLOMBIA', 'key', 'Miller', 'Kyle', 'sunil', 'NSHUT', 'Gonzalez', 'Albertville', 'export', 'Almeida', 'Newtown', 'per', 'Yoshikicyo', 'info', 'Sutton', 'Junichi', 'Please', 'attach', 'St', 'unexpected', 'Bob', 'Tamaela', 'City', 'Bishop', 'eine', 'Mohammed', 'owner', 'hi', 'melendez', 'epic', 'Augusto', '%', 'Heider', 'DeCooman', 'Huber', 'misqueue', 'Enviado', 'Ishak', 'often', 'Miguel', 'fri', 'Mon-Thu', 'Ruben', 'txt', 'fuller', 'Barragan', 'Nair', 'Katsumata', 'Hill', 'didn', 'manila', 'link', 'options', 'expire', 'Miki', 'PERM', 'haven', 'Vasu', 'descr', 'Danila', 'sincerely', 'São', 'Jie', 'id', 'Krevat', 'Delhi', 'Castaneda', 'Rudy', 'Leo', 'edina', 'Robert', 'order', 'Hanashiro', 'Xing', 'Budafoki', 'Scott', 'Lima', 'Boardtype', 'upload', 'home', 'value', 'Yanrui', 'Linecard', 'Nagar', 'notes', 'web', 'Chico', 'Derrick', 'Mustapha', 'Microloops', 'close', 'Arroyo', 'ioszr', 'Sendai', 'Yasuo', 'also', 'create', 'desk', 'MAINMEM', 'Ashwin', 'HW', 'Karl', 'Heraghty', 'Huang', 'Africa', 'director', 'size', 'St.', 'manager', 'port', 'States', 'business', 'asr', 'state', 'Utah', 'style', 'Lei\\Bo', 'Bethesda', 'Bringdown', 'such', 'Memphis', 'Max', 'internet', 'Sergio', 'kindly', 'job', 'Cisco', 'boston', 'Maazkhan', 'Cintron', 'customer', 'Martha', 'category', 'want', 'high', 'Cherian', 'ship', 'zip', 'get', 'make', 'save', 'Fumitoshi', 'Neal', 'contract', 'Parkway', 'G.', 'regardto', 'tool', 'Melendez', 'WA', 'XingYi', 'Waigani', 'like', 'closed', 'Aleksander', 'Feng', 'Li', 'Sims', 'srch', 'answers', 'thurs', 'E.', 'Peter', 'Jamack', 'M', 'PAYET', 'actually', 'de', 'from', 'Rick', 'user', 'Coria', 'no', 'share', 'Harris', 'Qayyum', 'Kara', 'unicom', 'template', 'program', 'here', 'DOS', 'Ontario', 'Ecuador', 'Ahmed', 'New', 'McLain', 'Murgolo', 'Carlson', 'Hummel', 'Ma', 'Arturo', 'ask', 'Lindawood', 'skill', 'Takahashi', 'dot', 'file', 'Cuando', 'Dean', 'KADOTA', 'Faisal', 'technical', 'done', 'Gokul', 'tac', 'Nicky', 'Sarath', 'Palii', 'Proto', 'summary', 'S.', 'helloadi', 'Rodrigo', 'com', 'test', 'Ki', 'kinds', 'Virginia', 'Eaton', 'page', 'history', 'sumeet', 'Gilagam', 'possible', 'attachment', 'HTOM', 'rma', 'Ping', 'TX', 'Poland', 'Huevertech', 'Singh', 'Country', 'product', 'htts', 'Shawn', 'Saludos', 'Ikupu', 'por', 'Wandratsch', 'Yao', 'Dimitri', 'busy', 'well', 'John', 'Perguntas', 'div', 'request', 'FAIRFIELD', 'Pasmurnov', 'china', 'Libya', 'Sumeet', 'see', 'company', 'role', 'Umesh', 'things', 'Huaibin', 'thu', 'ARP', 'MO', 'Ye', 'Moonyong', 'open', 'Somuri', 'Sweden', 'Fleitas', 'city', 'romeoville', 'Craig', 'have', 'Arias', 'Korea', 'Cust', 'Guinea', 'Mikhailov', 'Ligang', 'LEI', 'pri', 'mean', 'Dallas', 'confirm', ')', 'Sarjapur', 'Para', 'Hoag', 'in', 'Hao', 'Dheeraj', 'sub', '\\', 'Entre', 'Blackburn', 'yes', 'Gheevarghese', 'ESTILHADOUROS', 'delivered', 'Carr', 'Basant', 'Secteur', 'est', 'Kai', 'http', 'docs', 'show', 'Crashinfo', 'khtml', 'Raj', 'Brown', 'Ettwejiri', 'Eduardo', 'IL', 'Chad', 'legal', 'become', 'public', 'IZZI', 'Können', 'Choi', 'Guerra', '(', 'issue', 'Mendonça', 'LaFayette', 'middleboro', 'Evans', 'Lopes', 'Jeremy', 'this', 'complete', 'gbt', 'email', 'Peng', 'mycase', 'Ashville', 'Edward', 'Singapore', 'index', 'Amazon', 'Mansoor', 'Rivalino', 'engineer', 'overall', 'Tang', 'first', 'Miyagi', 'Denton', 'fine', 'Chung', 'Romer', 'afterhours', 'Dharmarajan', 'Marco', 'Yinyan', '#name', 'Christina', 'Australia', 'West', 'Ganesh', 'Prinkesh', 'ctc', 'chicago', 'Hofmann', 'Salvador', 'Dewan', 'system', 'Matecki', 'load', 'United', 'Hunyadi', 'current', 'type', 'Suresh', 'William', 'permission', 'Ward', 'Minsk', 'comment', 'take', 'Tokyo', 'POLAND', 'Kalbfeld', 'zero', 'Ireland', 'true', 'dispatch', 'Kate', 'reason', 'Bai', 'ORR', 'i', 'Brigitte', 'session', 'adam', 'attachments', 'hay', 'Kroehle', 'kara', 'Louis', 'Norway', 'Loganatha', 'january', 'refer', 'pst', 'problem', 'Christopher', 'himanshu', 'text', 'not', 'Yawson', 'status', 'Willingham', 'Kleinmachnow', 'Lui', 'Macau', 'Hokeun', 'Eric', 'Rd', 'Okrie', 'known', 'app', 'Richmond', 'regards', 'michigan', 'PARK', 'LA', 'Taiwan', 'Muthu', 'Naidoo', 'Zhu', 'Zeledon', 'Ravi', 'Gibson', 'ALFENA', 'Birmingham', 'Mr.', 'EDWARDSVILLE', 'Tx', 'Telikom', 'Shaurav', 'Low', 'replicate', 'TRIANGLE', 'FRANCE', 'color', 'Mack', 'Nunnally', 'Raziel', 'Anran', 'Masaya', 'Espero', 'Csaba', 'communication', 'HA', 'America', 'Spencer', 'arvind', 'This', 'safari', 'Hamburg', 'Liou', 'Le', 'Sequeira', 'webex', 'service', 'Shimraan', 'Whiteville', 'workaround', 'Robson', 'note', 'HONOLULU', 'Cassidy', 'Antoniades', 'india', 'focus', 'Maaz', 'Team', 'will', 'Belarus', 'priority', 'Shiyao', 'Jason', 'Hello', 'Griffin', 'Cynthia', 'Sipov', 'str', 'US', 'Durrani', 'since', 'Arvind', 'Core', 'Theresa', '|', 'Andreas', 'the', 'Matt', 'available', 'aim', 'jio', 'line', 'Wisconsin', 'Pack', 'addr', 'Dear', 'hey', 'Weiner', 'Asia', 'Allen', 'Peggy', 'is', 'Johnson', 'Ting', 'apac', 'Sultanova', 'Xun', 'Barreto', 'Mazur', 'adi', 'Shuang', 'hours', 'Azeem', 'automatic', 'Amphitheatre', 'huang', 'many', 'Gago', 'Al', 'Beijing', 'Bawa', 'Cary', 'Shunsuke', 'region', 'Dino', 'level', 'expect', 'solution', 'case', 'BRINGDOWN', 'Subramanian', 'utc', 'version', 'Poovalingam', 'Emma', 'Hungary', 'iii', 'max', 'Mei', 'Hu', 'Johnson-Rustvold', 'Kulvinder', 'knyazev', 'North', 'elmhurst', 'instructions', 'Oman', 'Pearse', 'defekte', 'step', 'support', 'cisco', 'Lalit', 'lead', 'Polisetty', 'provide', 'Jean', 'TAIPEI', 'cisco.com', 'title', 'pas', 'number', 'part', 'mozilla', 'Jharkand', 'Yin', 'ensure', 'U.S.', 'input', 'aisne', 'all', 'Condict', 'Elmar', 'Weerasekara', 'Denver', 'Tulsi', 'Gabriela', 'duplicate', 'Bhadauria', 'bug', 'Sudeep', 'Durelle', 'Aragon', 'Daniel', 'Yang', 'Austria', 'data', 'device', 'Burns', 'manually', 'shipent', 'Zhao', 'Karte', 'americas', 'Hi', 'J.', 'false', 'low', 'small', 'David', 'Stevens', 'Gustavo', 'detailasr903', 'unknown', 'Sanaullah', 'Cortez', 'additional', 'MR', 'location', 'process', 'Zeeshan', 'Alam', 'Borman', 'Joshua', 'Rawat', 'Stavanger', 'Huawen', 'Pandian', 'Villanueva', 'PORTUGAL', 'large', 'final', 'Hector', 'Chakravarti', 'Fei', 'cases', 'asa', 'tech', 'Iwamochi', 'Dominic', 'Katarzyna', 'Joe', 'Tama', 'Pang', 'exception', 'Victor', 'ROCHESTER', 'detail', 'Mora', 'Wenlong', 'GREENWOOD', 'group', 'Province', 'source', 'end', 'Prakashbabu', 'Wei', 'collect', 'tmr', 'Reseat', 'Hoerauf', 'Shang', 'Pepin', 'admin', 'Kim', 'Powers', 'Japan'}
lemma = WordNetLemmatizer()
#lemma = LancasterStemmer()
def clean_text(inputStr):
	y = []
	for word in inputStr:
		status = True
		for w in word.split(' '):
			if(lemma.lemmatize(w, get_pos(word)).lower() in stop or lemma.lemmatize(w, get_pos(word)).lower() in stoplist):
			#if(lemma.stem(w).lower() in stop or lemma.stem(w).lower() in stoplist):
				status = False
			if(w.isalpha() == False):
				#print(w)
				status = False
		if(status == True and len(word) > 2):
			#print(word)
			#print(word.isalpha())
			y.append(word.lower())
	return list(set(y))

def get_human_names(text):
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)
    person_list = []
    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        if len(person) > 1: #avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
        person = []
    return (person_list)


#For each PF, run the function
#pf_list = settings.get("SR_Source","pfList").split(',')
pf_list = ['ASR9000']
pf = 'ASR9000'

#for pf in pf_list:
asr_df = pd.read_csv("/auto/vgapps-cstg02-vapps/analytics/csap/ingestion/opfiles/srData/SRDetails_ASR9000_20180801_RMA_Merged.csv", encoding='utf-8') #180918SRData.csv
#asr_df = asr_df[asr_df['rma_count'] > 0]

'''
collection = db[collection_prefix + str(pf)]
print(collection)
cursor = collection.find({'rma_count': {'$gt': 0}})
asr_df = pd.DataFrame(list(cursor))
'''
asr_df["notesdata.notes_detail"].fillna("", inplace=True)
asr_df['kpi_sr_details.sr_resolution_code'].fillna("", inplace=True)
asr_df['kpi_sr_details.sr_underlying_cause_code'].fillna("", inplace=True)
asr_df['kpi_sr_details.sr_troubleshooting_description'].fillna("", inplace=True)
asr_df['kpi_sr_details.sr_problem_summary'].fillna("", inplace=True)
asr_df['kpi_sr_details.sr_underlying_cause_desc'].fillna("", inplace=True)

notes_type = ['RESOLUTION SUMMARY', 'PROBLEM DESCRIPTION', 'KT PROBLEM ANALYSIS', 'CUSTOMER SYMPTOM', 'CASE REVIEW', 'Problem Description', 'Other', 'Case Review', 'REQUE REASON']
notes_type = ['Email In', 'EMAIL IN']
asr_df = asr_df[asr_df['notesdata.note_type'].isin(notes_type)]
asr_df = asr_df.groupby('kpi_sr_details.sr_number').agg({'kpi_sr_details.sr_number':'first', 'notesdata.notes_detail': ' '.join, 'kpi_sr_details.sr_hw_product_erp_family':'first', 'kpi_sr_details.sr_create_timestamp':'first', 'kpi_sr_details.sr_defect_number':'first', 'kpi_sr_details.sr_resolution_code':'first', 'kpi_sr_details.sr_underlying_cause_code': 'first', 'kpi_sr_details.sr_underlying_cause_desc': 'first', 'kpi_sr_details.sr_problem_summary': 'first', 'kpi_sr_details.sr_troubleshooting_description': 'first'})

#notes = asr_df["notes"]+'. '+ asr_df['sr_resolution_code']+ '. '+ asr_df['sr_underlying_cause_code']+'. '+ asr_df['sr_troubleshooting_description']+ '. '+ asr_df['sr_problem_summary'] + '. ' + asr_df['sr_underlying_cause_desc']
notes = asr_df["notesdata.notes_detail"]
notes = notes.replace(np.nan, '', regex=True)

##########Preprocessing###########
'''
k=0
for text in notes:
	print(k)
	k = k + 1
	for sent in nltk.sent_tokenize(text):
		tokens = nltk.tokenize.word_tokenize(sent)
		tags = st.tag(tokens)
		#print(tags)
		for tag in tags:
			if tag[1]=='PERSON' or tag[1]=='LOCATION': stoplist.add(tag[0])
			#if tag[1]=='ORGANIZATION': print(tag)
'''

k = 0
all_names = []
for text in notes:
	k = k + 1
	print(k)
	anames = get_human_names(text)
	for name in names: 
		all_names.append((HumanName(name).last).lower())
		all_names.append((HumanName(name).first).lower())

stoplist = set(list(stoplist) + all_names)

c=0
new_notes = []
for note in notes:
	c+=1
	regex = r"\S*@\S*\s?"
	note = re.sub(regex, '', note, 0)
	stop_words=['From','To','Case ID',':', 'Subject', 'Part', 'Cc', 'SR', 'CASE', 'STATUS', 'CASE SUBJECT', 'PRIORITY', 'IMPORTANT', 'NAME']
	for word in stop_words:
		if word in note:
			note=note.replace(word,"")
	note = re.sub(' +',' ',note)
	regex = r"\S* # \S*\s?"
	note = re.sub(regex, '', note, 0)
	regex = r"\S*# \S*\s?"
	note = re.sub(regex, '', note, 0)
	regex = r'\w*[0-9]\w*'
	note = re.sub(regex, '', note, 0)
	#remove dates and time
	regex = r"\d\d-\S*-\d\d\d\d*"
	note = re.sub(regex, '', note, 0)
	note = re.sub(' +',' ',note)
	new_notes.append(note)

print("Completed intial preprocessing")
asr_df["preprocessed_notes"] = new_notes

asr_df["preprocessed_notes"] = asr_df["preprocessed_notes"].fillna('')

docs_complete = asr_df["preprocessed_notes"].tolist()

#Consider Problem Details 		
imp_desc = []
for doc in docs_complete:
	#s = '\nProblem Details Currently MSN pair connected by only 1 of 2 links. Cannot determine why link will not synchronize.\n'
	result = re.search('Problem Details (.*)\n', doc)
	result2 = re.search('Problem Description (.*)\n', doc)
	result3 = re.search('Problem Summary (.*)\n', doc)
	status = True
	r = ''
	if(result):
		r = result.group(1)
		status = False
	if(result2):
		r = r + result2.group(1)
		status = False
	if(result3):
		r = r + result3.group(1)
		status = False
	if(status == True):
		imp_desc.append('')
	else:
		imp_desc.append(r)
	if(status==False):
		print(status)


imp_docs_processed=[]
for doc in imp_desc:
	doc = doc.replace('*', ' ')
	doc = doc.replace('=', ' ')
	doc = doc.replace('__', ' ')
	doc = doc.replace('..', ' ')
	doc = doc.replace(';', ' ')
	doc = doc.replace('"', ' ')
	doc = doc.replace(r'[^\x00-\x7F]+', '')
	doc = doc.replace('(', ' ')
	doc = doc.replace(')', ' ')
	doc = doc.replace('[', ' ')
	doc = doc.replace(']', ' ')
	doc = doc.replace('.', ' ')
	doc = doc.replace('\r', ' ')
	doc = doc.replace('+', ' ')
	doc = doc.replace('|', ' ')
	doc = doc.replace('#', ' ')
	doc = re.sub(r"==", " ", doc, 0)
	doc = re.sub(r"-", " ", doc, 0)
	doc = re.sub(r"&", " ", doc, 0)
	doc = re.sub(r"--", "", doc, 0)
	doc = re.sub(r"{", " ", doc, 0)
	doc = re.sub(r"}", " ", doc, 0)
	doc = re.sub(r":", " ", doc, 0)
	doc = re.sub(r"/", " ", doc, 0)
	doc = re.sub(r">", " ", doc, 0)
	doc = re.sub(r"<", " ", doc, 0)
	doc = re.sub(r",", " ", doc, 0)
	doc = re.sub(r"'", " ", doc, 0)
	doc = re.sub(r"!", " ", doc, 0)
	doc = re.sub(r"@", " ", doc, 0)
	doc = re.sub(r"GMT", " ", doc, 0)
	imp_docs_processed.append(doc)

def get_pos( word ):
    w_synsets = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len(  [ item for item in w_synsets if item.pos()=="n"]  )
    pos_counts["v"] = len(  [ item for item in w_synsets if item.pos()=="v"]  )
    pos_counts["a"] = len(  [ item for item in w_synsets if item.pos()=="a"]  )
    pos_counts["r"] = len(  [ item for item in w_synsets if item.pos()=="r"]  )
    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]


#We got strings
#From these we need to remove stopwords and add this to keywords list
imp_stoplist = {'descr', 'you', 'notice', 'replace', 'how', 'any', 'hesitate', 'worldwide', 'direct', 'korean', 'justification', 'best', 'churn', 'feb', 'only', 'david', 'dumps', 'dump', 'purpose', 'purposes', 'pkt', 'label', 'ieee', 'start', 'bad', 'initiate', 'recent', 'jul', 'event', 'budafoki', 'but', 'up.', 'init', 'paul', 'project', 'projects', 'threshold', 'engineer', 'engineering', 'æœºç®±å‡çº§standby', 'xinyuan', 'municipal', 'financial', 'kompetencia', 'jtæ–°æ±äº¬', 'sep', 'onsite', 'kiss', 'into', 'kÃ¡lmÃ¡n', 'Ã¡rpÃ¡d', 'ã‚·ãƒªã‚¢ãƒ«ç•ªå·', 'august', 'date', 'confirmed', 'warning', 'directory', 'set', 'sho', 'dir', 'instal', 'main', 'none', 'count', 'now', 'although', 'clear', 'sesiÃ³n', 'atenciÃ³n', 'set', 'ingenieros', 'cuando', 'sesiones', 'msgq', 'faxç•ªå·', 'non', 'iosxr', 'july', 'alarm', 'cold', 'waiting', 'usage', 'ordered', 'filter', 'down', 'richmond', 'pass', 'try', 'don', 'digital', 'full', 'joÃ£o', 'mike', 'dallas', 'louis', 'continue', 'our', 'via', 'what', 'tac', 'nodes', 'follow', 'one', 'both', 'sit', 'root', 'cause', 'shane', 'below', 'attached', 'they', 'merritt', 'mobile', 'laptop', 'device', 'kishiki', 'saitama', 'currently', 'available', 'profile', 'default', 'status', 'dev', 'prd', 'kill', 'killing', 'exit', 'enter', 'previous', 'nodeid', 'id', 'jrdhqiaf', 'eta', 'ordernumber', 'present', 'report', 'reports', 'asking', 'ask', 'new', 'newly', 'old', 'louis', 'beijing', 'agreed', 'ready', 'rochester', 'sat', 'sun', 'aug', 'mon', 'tue', 'web', 'jan', 'mar', 'april', 'june', 'jun', 'douglas', 'single', 'multiple', 'tired', 'condition', 'instance', 'host', 'console', 'closing', 'debug', 'hello', 'macintosh', 'Jeffrey', 'Rumana', 'time', 'private', 'need', 'ios', 'Grettel', 'Site', 'ushaditya', 'requeue', 'André', 'Jake', 'Englewood', 'Aurora', 'Danilo', 'Simmons', 'Phone', 'details', 'Hays', 'code', 'action', 'arias', 'as', 'Dhan', 'automatically', 'Tim', 'Sood', 'verify', 'Rommon', 'Logan', 'Mike', 'Xiaodong', 'sr', 'Doug', 'brief', 'amazon', 'can', 'WARSAW', 'JAPAN', 'Coutinho', 'Srikant', 'Kevin', 'Pls', 'Ferenc', 'Kathleen', 'Fan', 'Oklahoma', 'sandy', 'contributor', 'Thoms', 'jim', 'windows', 'Makoto', 'Ricardo', 'Jalet', 'country', 'USA', 'Toth', 'please', 'people', 'to', 'initial', 'Chen', 'Htom', 'manual', 'Rua', 'Andre', 'Yamamoto', 'asr903', 'Harada', 'that', 'angel', 'workshop', 'Aravind', 'List', 'Yohei', 'Leonel', 'Hong', 'Seoul', 'operation', 'Thiago', 'Alexey', 'download', 'SAN', 'look', 'Juan', 'Port', 'Figueira', 'Sharma', 'poland', 'johnson', 'Waldemar', 'Hannam', 'Halsey', 'Vanessa', 'sure', 'Wien', 'en', 'Chris', 'Haneef', 'Canada', 'crs', 'click', 'Raghu', 'Manila', 'Jim', 'management', 'Merritt', 'contact', 'help', 'Varthur', 'Stefan', 'Jiang', 'Pacific', 'person', 'Tshoot', 'know', 'Webb', 'Dolores', 'Hou', 'Harsh', 'be', 'Silva', 'satellite', 'Papua', 'Sousa', 'Erwin', 'Shane', 'am', 'Loopback', 'Kadubeesanahalli', 'example', 'Cengiz', 'Andrew', 'sales', 'error', 'Type', 'Slidell', 'logs', 'Rodrigues', 'katarzyna', 'Prakash', 'employee', 'miguel', 'Ahmad', 'Pd', ';', 'Michigan', 'Ln', 'Shafi', 'Amanda', 'Pei', 'Guillermo', 'every', 'India', 'Ko', 'Heffernan', 'Lineside', 'Zhang', 'Hobli', 'address', 'alert', 'Risky', 'Joao', 'Goutham', 'trap', 'Patel', 'query', 'll', 'use', 'Hua', 'give', 'Carpenter', 'SE', 'safe', 'Slidel', 'Roseville', 'yet', 'Sunil', 'keep', 'for', 'Guangzhou', 'Charles', 'BUSADRERR', 'sood', 'Wang', 'SAITAMA', 'name', 'Ishwar', 'call', 'Vyatskaya', 'timestamp', 'Shelfmgr', 'SHENZHEN', 'Yi', 'with', 'phythian', 'Harddisk', 'KHTML', 'reddy', 'Samir', 'Mode', 'Hola', 'iox', 'Ochi', 'ST', 'Galiev', 'https', 'Luis', 'Dechant', 'Alfaro', 'queue', 'plan', 'when', 'Pandav', 'Hirokazu', 'reply', 'php', 'Tian', 'Didier', 'Bill', 'deliver', 'Fuller', '\x08', 'account', 'Rendon', 'Minghai', 'Janos', 'Alser', 'mac', 'Requeue', 'county', 'find', 'Haider', 'Kazuma', 'Jay', 'Morgan', 'Almirante', '.', 'impact', 'Ronnie', 'side', 'list', 'Moscow', 'Wanxin', 'shipment', 'errors', 'day', 'Amol', 'Angel', 'Cole', 'thanks', 'standard', 'equipo', 'charleston', 'BANGALORE', 'Sendhil', 'California', 'Ningbo', 'Nome', 'BUDAPEST', 'Linz', 'LIANG', 'Morobe', 'Douglas', 'last', 'would', 'Aliya', 'Acosta', 'AZ', 'Yue', 'Muthusamy', 'Vonda', 'Yan', 'Zheng', 'Kohlmann', 'H', 'Eva', 'change', 'zr', 'object', 'Firmensitz', 'Nord', 'Masaki', 'Russ', 'Cho', 'access', 'Marathalli', 'Dian', 'sale', 'Krishna', 'Mark', 'Xu', 'Lisowski', 'Berlin', 'Garren', 'Matthew', 'RTP', 'Luiz', 'Saitama', 'answer', 'way', 'require', 'FW', 'a', 'google', 'mazur', 'new', 'Lawrenceville', 'L', 'Kris', 'wish', 'Dores', 'NVRAM', 'Mohan', 'Kumar', 'Erero', 'Stemmons', 'ext', 'João', 'description', 'Tencent', 'Elodie', 'timely', 'Ritzl', 'Street', 'Daniels', 'Michael', 'check', 'Pedro', 'and', 'Shubham', 'Pete', 'systems', 'profile', 'mon', 'Haluza', 'local', 'James', 'greet', 'Santhana', 'ISP', 'Frank', 'Email', 'receive', 'think', 'Thatipalli', 'Toronto', 'work', 'Adi', 'thank', 'Darrell', 'Jacomini', 'Aguilar', 'tue', 'Safi', 'El', 'helpdesk', 'sev', 'rtp', 'browser', 'contribute', 'NC', 'come', 'Brian', 'ok', 'Hellos', 'MR.', 'STOCKTON', 'function', 'www', 'creation', 'Walnut', 'Paulo', 'install', 'different', 'chrome', 'california', 'Morales', 'workgroup', 'Duke', 'Nguyen', 'View', 'cust', 'something', 'cancel', 'PoE', 'Costa', 'Zok', 'hannam', 'Hsieh', 'Mohd', 'Nan', 'regard', 'insert', 'html', 'node', 'Gabriel', 'unkown', 'Pearl', 'application', 'Lu', 'Mililani', 'reliance', 'confirmation', 'illegal', 'Dorenkamp', 'fileid', 'Charlotte', 'Longfei', 'Idogawa', 'Bauer', 'Norriton', 'Xiangbo', 'free', 'Gilks', 'william', 'Sartori', 'us', 'Babu', 'Aguero', 'Dave', 'Dion', 'Sydney', 'P', 'subtechnology', 'Boyd', 'Kalman', 'michael', 'log', 'UFA', 'LC', 'Gerardo', 'Dresch', 'Lambrechts', 'span', 'Mr', 'february', 'Budapest', 'Hugo', 'store', 'luther', 'Europe', 'Masahiro', 'China', 'Bokaro', 'Richard', 'Endo', 'Ford', 'Carlos', 'Khairi', 'ist', 'good', 'definitions', '&', 'Mariaux', 'Omiya', 'Newark', 'content', 'VSC', 'HongGil', 'RSP0', 'Greenville', 'wikt', 'Tony', 'Three', 'america', 'Vitaly', 'Paul', 'Watts', 'kind', 'pid', 'Mountain', 'severity', 'Sie', 'Akriti', 'ganesh', 'M.S', 'subject', 'Shenzhen', 'japan', 'Vestal', 'ALAMOSA', 'Rodriguez', 'other', 'Ce', 'doc', 'Balakrishnan', 'DUBAI', 'Field', 'Milovan', 'ENOC', 'sort', 'been', 'York', 'Liu', 'Rafael', 'BOUCA', 'Wu', 'usually', 'team', 'Shota', 'Point', 'important', 'output', 'say', 'Lake', 'Jid', 'Kendal', 'Haug', 'Xie', 'active', 'include', 'South', 'Salamanderstraße', 'Haifeng', 'Mingyao', 'font', 'Castro', 'Aschenbrenner', 'Olsen', 'Miss', 'Pincode', 'colour', 'may', 'stanley', 'Luciano', 'amer', 'phone', 'search', 'update', 'Mais', 'Parc', 'same', 'Cao', 'weerasekara', 'wed', 'serial', 'dear', 'Salt', 'Rich', 'Hsueh', 'option', 'Arthur', 'Abdulhadi', 'cc', 'Leal', 'response', 'Sridhar', 'gmt', 'mgmt', 'Garcia', 'Kong', 'notification', 'Str', 'COLOMBIA', 'key', 'Miller', 'Kyle', 'sunil', 'NSHUT', 'Gonzalez', 'Albertville', 'export', 'Almeida', 'Newtown', 'per', 'Yoshikicyo', 'info', 'Sutton', 'Junichi', 'Please', 'attach', 'St', 'unexpected', 'Bob', 'Tamaela', 'City', 'Bishop', 'eine', 'Mohammed', 'owner', 'hi', 'melendez', 'epic', 'Augusto', '%', 'Heider', 'DeCooman', 'Huber', 'misqueue', 'Enviado', 'Ishak', 'often', 'Miguel', 'fri', 'Mon-Thu', 'Ruben', 'txt', 'fuller', 'Barragan', 'Nair', 'Katsumata', 'Hill', 'didn', 'manila', 'link', 'options', 'expire', 'Miki', 'PERM', 'haven', 'Vasu', 'descr', 'Danila', 'sincerely', 'São', 'Jie', 'id', 'Krevat', 'Delhi', 'Castaneda', 'Rudy', 'Leo', 'edina', 'Robert', 'order', 'Hanashiro', 'Xing', 'Budafoki', 'Scott', 'Lima', 'Boardtype', 'upload', 'home', 'value', 'Yanrui', 'Linecard', 'Nagar', 'notes', 'web', 'Chico', 'Derrick', 'Mustapha', 'Microloops', 'close', 'Arroyo', 'ioszr', 'Sendai', 'Yasuo', 'also', 'create', 'desk', 'MAINMEM', 'Ashwin', 'HW', 'Karl', 'Heraghty', 'Huang', 'Africa', 'director', 'size', 'St.', 'manager', 'port', 'States', 'business', 'asr', 'state', 'Utah', 'style', 'Lei\\Bo', 'Bethesda', 'Bringdown', 'such', 'Memphis', 'Max', 'internet', 'Sergio', 'kindly', 'job', 'Cisco', 'boston', 'Maazkhan', 'Cintron', 'customer', 'Martha', 'category', 'want', 'high', 'Cherian', 'ship', 'zip', 'get', 'make', 'save', 'Fumitoshi', 'Neal', 'contract', 'Parkway', 'G.', 'regardto', 'tool', 'Melendez', 'WA', 'XingYi', 'Waigani', 'like', 'closed', 'Aleksander', 'Feng', 'Li', 'Sims', 'srch', 'answers', 'thurs', 'E.', 'Peter', 'Jamack', 'M', 'PAYET', 'actually', 'de', 'from', 'Rick', 'user', 'Coria', 'no', 'share', 'Harris', 'Qayyum', 'Kara', 'unicom', 'template', 'program', 'here', 'DOS', 'Ontario', 'Ecuador', 'Ahmed', 'New', 'McLain', 'Murgolo', 'Carlson', 'Hummel', 'Ma', 'Arturo', 'ask', 'Lindawood', 'skill', 'Takahashi', 'dot', 'file', 'Cuando', 'Dean', 'KADOTA', 'Faisal', 'technical', 'done', 'Gokul', 'tac', 'Nicky', 'Sarath', 'Palii', 'Proto', 'summary', 'S.', 'helloadi', 'Rodrigo', 'com', 'test', 'Ki', 'kinds', 'Virginia', 'Eaton', 'page', 'history', 'sumeet', 'Gilagam', 'possible', 'attachment', 'HTOM', 'rma', 'Ping', 'TX', 'Poland', 'Huevertech', 'Singh', 'Country', 'product', 'htts', 'Shawn', 'Saludos', 'Ikupu', 'por', 'Wandratsch', 'Yao', 'Dimitri', 'busy', 'well', 'John', 'Perguntas', 'div', 'request', 'FAIRFIELD', 'Pasmurnov', 'china', 'Libya', 'Sumeet', 'see', 'company', 'role', 'Umesh', 'things', 'Huaibin', 'thu', 'ARP', 'MO', 'Ye', 'Moonyong', 'open', 'Somuri', 'Sweden', 'Fleitas', 'city', 'romeoville', 'Craig', 'have', 'Arias', 'Korea', 'Cust', 'Guinea', 'Mikhailov', 'Ligang', 'LEI', 'pri', 'mean', 'Dallas', 'confirm', ')', 'Sarjapur', 'Para', 'Hoag', 'in', 'Hao', 'Dheeraj', 'sub', '\\', 'Entre', 'Blackburn', 'yes', 'Gheevarghese', 'ESTILHADOUROS', 'delivered', 'Carr', 'Basant', 'Secteur', 'est', 'Kai', 'http', 'docs', 'show', 'Crashinfo', 'khtml', 'Raj', 'Brown', 'Ettwejiri', 'Eduardo', 'IL', 'Chad', 'legal', 'become', 'public', 'IZZI', 'Können', 'Choi', 'Guerra', '(', 'issue', 'Mendonça', 'LaFayette', 'middleboro', 'Evans', 'Lopes', 'Jeremy', 'this', 'complete', 'gbt', 'email', 'Peng', 'mycase', 'Ashville', 'Edward', 'Singapore', 'index', 'Amazon', 'Mansoor', 'Rivalino', 'engineer', 'overall', 'Tang', 'first', 'Miyagi', 'Denton', 'fine', 'Chung', 'Romer', 'afterhours', 'Dharmarajan', 'Marco', 'Yinyan', '#name', 'Christina', 'Australia', 'West', 'Ganesh', 'Prinkesh', 'ctc', 'chicago', 'Hofmann', 'Salvador', 'Dewan', 'system', 'Matecki', 'load', 'United', 'Hunyadi', 'current', 'type', 'Suresh', 'William', 'permission', 'Ward', 'Minsk', 'comment', 'take', 'Tokyo', 'POLAND', 'Kalbfeld', 'zero', 'Ireland', 'true', 'dispatch', 'Kate', 'reason', 'Bai', 'ORR', 'i', 'Brigitte', 'session', 'adam', 'attachments', 'hay', 'Kroehle', 'kara', 'Louis', 'Norway', 'Loganatha', 'january', 'refer', 'pst', 'problem', 'Christopher', 'himanshu', 'text', 'not', 'Yawson', 'status', 'Willingham', 'Kleinmachnow', 'Lui', 'Macau', 'Hokeun', 'Eric', 'Rd', 'Okrie', 'known', 'app', 'Richmond', 'regards', 'michigan', 'PARK', 'LA', 'Taiwan', 'Muthu', 'Naidoo', 'Zhu', 'Zeledon', 'Ravi', 'Gibson', 'ALFENA', 'Birmingham', 'Mr.', 'EDWARDSVILLE', 'Tx', 'Telikom', 'Shaurav', 'Low', 'replicate', 'TRIANGLE', 'FRANCE', 'color', 'Mack', 'Nunnally', 'Raziel', 'Anran', 'Masaya', 'Espero', 'Csaba', 'communication', 'HA', 'America', 'Spencer', 'arvind', 'This', 'safari', 'Hamburg', 'Liou', 'Le', 'Sequeira', 'webex', 'service', 'Shimraan', 'Whiteville', 'workaround', 'Robson', 'note', 'HONOLULU', 'Cassidy', 'Antoniades', 'india', 'focus', 'Maaz', 'Team', 'will', 'Belarus', 'priority', 'Shiyao', 'Jason', 'Hello', 'Griffin', 'Cynthia', 'Sipov', 'str', 'US', 'Durrani', 'since', 'Arvind', 'Core', 'Theresa', '|', 'Andreas', 'the', 'Matt', 'available', 'aim', 'jio', 'line', 'Wisconsin', 'Pack', 'addr', 'Dear', 'hey', 'Weiner', 'Asia', 'Allen', 'Peggy', 'is', 'Johnson', 'Ting', 'apac', 'Sultanova', 'Xun', 'Barreto', 'Mazur', 'adi', 'Shuang', 'hours', 'Azeem', 'automatic', 'Amphitheatre', 'huang', 'many', 'Gago', 'Al', 'Beijing', 'Bawa', 'Cary', 'Shunsuke', 'region', 'Dino', 'level', 'expect', 'solution', 'case', 'BRINGDOWN', 'Subramanian', 'utc', 'version', 'Poovalingam', 'Emma', 'Hungary', 'iii', 'max', 'Mei', 'Hu', 'Johnson-Rustvold', 'Kulvinder', 'knyazev', 'North', 'elmhurst', 'instructions', 'Oman', 'Pearse', 'defekte', 'step', 'support', 'cisco', 'Lalit', 'lead', 'Polisetty', 'provide', 'Jean', 'TAIPEI', 'cisco.com', 'title', 'pas', 'number', 'part', 'mozilla', 'Jharkand', 'Yin', 'ensure', 'U.S.', 'input', 'aisne', 'all', 'Condict', 'Elmar', 'Weerasekara', 'Denver', 'Tulsi', 'Gabriela', 'duplicate', 'Bhadauria', 'bug', 'Sudeep', 'Durelle', 'Aragon', 'Daniel', 'Yang', 'Austria', 'data', 'device', 'Burns', 'manually', 'shipent', 'Zhao', 'Karte', 'americas', 'Hi', 'J.', 'false', 'low', 'small', 'David', 'Stevens', 'Gustavo', 'detailasr903', 'unknown', 'Sanaullah', 'Cortez', 'additional', 'MR', 'location', 'process', 'Zeeshan', 'Alam', 'Borman', 'Joshua', 'Rawat', 'Stavanger', 'Huawen', 'Pandian', 'Villanueva', 'PORTUGAL', 'large', 'final', 'Hector', 'Chakravarti', 'Fei', 'cases', 'asa', 'tech', 'Iwamochi', 'Dominic', 'Katarzyna', 'Joe', 'Tama', 'Pang', 'exception', 'Victor', 'ROCHESTER', 'detail', 'Mora', 'Wenlong', 'GREENWOOD', 'group', 'Province', 'source', 'end', 'Prakashbabu', 'Wei', 'collect', 'tmr', 'Reseat', 'Hoerauf', 'Shang', 'Pepin', 'admin', 'Kim', 'Powers', 'Japan'}
imp_stoplist = set(list(imp_stoplist) + list(stoplist))
lemma = WordNetLemmatizer()
def imp_clean_text(inputStr):
    normalized = " ".join(lemma.lemmatize(word, get_pos(word)) for word in str(inputStr).lower().split())
    stop_free = " ".join([i for i in str(normalized).lower().split() if i not in (stop and imp_stoplist)])
    normalized = " ".join(lemma.lemmatize(word, get_pos(word)) for word in stop_free.split())
    x = normalized.split()
    y = [s for s in x if len(s) > 2]
    y = " ".join(y)
    return y

imp_docs_clean = [imp_clean_text(doc) for doc in imp_docs_processed]


docs_processed=[]
for doc in docs_complete:
	doc = doc.replace('*', ' ')
	doc = doc.replace('=', ' ')
	doc = doc.replace('__', ' ')
	doc = doc.replace('..', ' ')
	doc = doc.replace(';', ' ')
	doc = doc.replace('"', ' ')
	doc = doc.replace(r'[^\x00-\x7F]+', '')
	doc = doc.replace('(', ' ')
	doc = doc.replace(')', ' ')
	doc = doc.replace('[', ' ')
	doc = doc.replace(']', ' ')
	doc = doc.replace('+', ' ')
	doc = doc.replace('|', ' ')
	doc = doc.replace('#', ' ')
	doc = re.sub(r"==", " ", doc, 0)
	doc = re.sub(r"-", " ", doc, 0)
	doc = re.sub(r"&", " ", doc, 0)
	doc = re.sub(r"--", "", doc, 0)
	doc = re.sub(r"{", " ", doc, 0)
	doc = re.sub(r"}", " ", doc, 0)
	doc = re.sub(r":", " ", doc, 0)
	doc = re.sub(r"/", " ", doc, 0)
	doc = re.sub(r">", " ", doc, 0)
	doc = re.sub(r"<", " ", doc, 0)
	doc = re.sub(r",", " ", doc, 0)
	doc = re.sub(r"'", " ", doc, 0)
	doc = re.sub(r"!", " ", doc, 0)
	doc = re.sub(r"@", " ", doc, 0)
	doc = re.sub(r"GMT", " ", doc, 0)
	docs_processed.append(doc)

c=0
final_docs=[]
for doc in docs_processed:
	print(c)
	np_extractor = NPExtractor(doc)
	result = np_extractor.extract()
	#final_docs.append(" ".join(result))
	final_docs.append(result)
	c=c+1


print("Completed keyword extractions")
asr_df["final_notes"] = final_docs

docs_clean = [clean_text(doc) for doc in asr_df["final_notes"]]

#Merge imp_docs_clean and docs_clean
final_keywords = []
for i in range(0, len(docs_clean)):
	if(len(imp_docs_clean[i]) > 0):
		final_keywords.append([imp_docs_clean[i]])
		final_keywords[i] = final_keywords[i] + docs_clean[i]
	else:
		final_keywords.append(docs_clean[i])


asr_df['keywords'] = final_keywords
#asr_df = asr_df.groupby('sr_number').agg({'sr_number':'first', 'keywords': ' '.join, 'sr_hw_product_erp_family':'first', 'sr_create_timestamp':'first', 'sr_defect_number':'first', 'sr_resolution_code':'first', 'sr_underlying_cause_code': 'first', 'sr_underlying_cause_desc': 'first', 'sr_problem_summary': 'first', 'sr_troubleshooting_description': 'first'})

asr_df.to_csv("fa_srNotes_all" + pf + "_processed_2.csv", encoding='utf-8')

final_df = pd.DataFrame()
final_df['sr_number'] = asr_df['kpi_sr_details.sr_number']
final_df['notes_detail'] = asr_df['notesdata.notes_detail']
final_df['sr_hw_product_erp_family'] = asr_df['kpi_sr_details.sr_hw_product_erp_family']
final_df['sr_create_timestamp'] = asr_df['kpi_sr_details.sr_create_timestamp']
final_df['sr_defect_number'] = asr_df['kpi_sr_details.sr_defect_number']
final_df['sr_resolution_code'] = asr_df['kpi_sr_details.sr_resolution_code']
final_df['sr_underlying_cause_code'] = asr_df['kpi_sr_details.sr_underlying_cause_code']
final_df['sr_underlying_cause_desc'] = asr_df['kpi_sr_details.sr_underlying_cause_desc']
final_df['sr_problem_summary'] = asr_df['kpi_sr_details.sr_problem_summary']
final_df['sr_troubleshooting_description'] = asr_df['kpi_sr_details.sr_troubleshooting_description']
final_df['preprocessed_notes'] = asr_df['preprocessed_notes']
final_df['final_notes'] = asr_df['final_notes']
final_df['keywords'] = asr_df['keywords']

######LDA modeling to find the topics and then predict the topic######
#Building the documents dataset
list_keywords = list(final_df['keywords'])
keywords_lst = []
for i in list_keywords:
	i = " ".join(i)
	#Do stemming and append stemmed words only
	#new_string = ' '.join([w for w in i.split() if len(w)>2])
	new_string = ' '.join([lemma_stemmer.stem(w) for w in i.split() if len(w)>2])
	keywords_lst.append(new_string.split(' '))


dictionary = corpora.Dictionary(keywords_lst)

#dictionary.filter_extremes(no_below=4, no_above=0.4)
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in keywords_lst]

print("Running the LDA model")
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary)#, passes=50, iterations=500)
print("Saving the LDA model")

topics_words = ldamodel.print_topics(num_topics=50, num_words = 200)
c=0
doc_topics=[]
prob=[]
for doc in doc_term_matrix:
	a = sorted(ldamodel[doc], key=lambda x: x[1])[-1]
	doc_topics.append(a[0])
	prob.append(a[1])
	c=c+1

print("DataFrame created")
final_df["Topic_number"] = doc_topics
final_df["topic_probability"] = prob

records = json.loads(final_df.T.to_json()).values()
db.ASR9000_all_keywords_email.insert(records)



topics_matrix = ldamodel.show_topics(formatted=False,num_words=200, num_topics=10)
topics_matrix = np.array((topics_matrix),dtype=list)
topics_df = pd.DataFrame()
top_probs = []
top_words = []
top = []
for topic in range(0, 10):
	a = topics_matrix[topic]
	for i in range(0,200):
		top.append(topic)
		top_words.append(a[1][i][0])
		top_probs.append(a[1][i][1])

topics_df['Topic_number'] = top
topics_df['keyword'] = top_words
topics_df['probability'] = top_probs
topics_df['PF'] = pf

records = json.loads(topics_df.T.to_json()).values()
#db.SR_topic_keywords.drop()
db.ASR9000_FA_topics_email.insert(records)


#######Mapping of Topic number - SRs in it and keywords########
collection = db['ASR9000_FA_topics_email']
cursor = collection.find({}) # query
topics_df =  pd.DataFrame(list(cursor))

topics_df1 = topics_df
topics_df1.probability = topics_df1.probability.astype(str)
topics_df1 = topics_df1.groupby('Topic_number').agg({'Topic_number':'first', 'keyword': ','.join, 'PF':'first'})

collection = db['ASR9000_all_keywords_email']
cursor = collection.find({}) # query
final_df =  pd.DataFrame(list(cursor))

final_df1 = final_df
final_df1.sr_number = final_df1.sr_number.astype(str)
final_df1 = final_df1.groupby('Topic_number').agg({'Topic_number':'first', 'sr_number': ','.join})

topics_df1['sr_number'] = final_df1['sr_number']

records = json.loads(topics_df1.T.to_json()).values()
db.ASR9000_FA_SR_email.insert(records)
