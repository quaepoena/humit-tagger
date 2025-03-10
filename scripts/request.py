import requests
import urllib.parse
import sys

UD2DRS_SERVER_URL_PREFIX='http://127.0.0.1:6000/'

content=open(sys.argv[1],"r").read()

x=requests.post(UD2DRS_SERVER_URL_PREFIX, {'lang':'bm','text':content, 'format':"json"})
print(x.text)
