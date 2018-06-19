import requests
from requests.auth import HTTPBasicAuth
import AUTH
import code
import pandas as pd

def get_vault_data():
	r=requests.get("http://vault.elucd.com/news/",auth=HTTPBasicAuth(AUTH.username, AUTH.password))
	df=pd.DataFrame(r.json())
	return df
df=get_vault_data()
code.interact(local=locals())