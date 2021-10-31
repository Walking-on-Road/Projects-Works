#UA：User-Agent（请求载体的身份标识）
#UA检测：门户网站的服务器会检测对应请求的载体身份标识，如果检测到请求的载体身份标识为某一款浏览器，
#说明该请求是一个正常的请求。但是，如果检测到请求的载体身份标识不是基于某一款浏览器的，则表示该请求
#为不正常的请求（爬虫），则服务器端就很有可能拒绝该次请求。

#UA伪装：让爬虫对应的请求载体身份标识伪装成某一款浏览器
import requests
from lxml import etree
import pandas as pd
import json

############
session = requests.Session()
headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36 Edg/94.0.992.38'
    }
login_url = "https://insight.shrp2nds.us/j_spring_security_check" #网站地址
data = {
        "j_username": "", #登录账号
        "j_password": "", #登录密码
}
response = session.post(url=login_url,headers=headers,data=data)
print(response.status_code)

detail_data_list = []
detail_url = 'https://insight.shrp2nds.us/dataset/data/38'
for page in range(1,168,1):
    page = str(page)
    data = {
        "page": page,
        "rows": "250",
        "sidx": "1254",
        "sord": "asc",
        "filters": "null",
        }
    json_ids = session.post(url=detail_url, headers=headers, data=data).json()
    detail_data_list.append(json_ids["rows"])
print(detail_data_list)
