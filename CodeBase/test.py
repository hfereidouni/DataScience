import json
import os
import pandas

with open('./nhl_data/2016020001.json') as f:
    q=json.load(f)
    print(q)
