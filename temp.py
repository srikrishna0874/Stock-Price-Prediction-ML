# Python 3
import http.client
import urllib.parse
import json

conn = http.client.HTTPSConnection('api.marketaux.com')

params = urllib.parse.urlencode({
    'api_token': 'PLeOSP7c0Yl1pyrP0me9jKNm5BZ25E9F6WpTvq2G',
    'symbols': 'AAPL',
    'limit': 5,
})

conn.request('GET', '/v1/news/all?{}'.format(params))

res = conn.getresponse()
data = res.read()
data = data.decode()

json_object = json.loads(data)
#for i in range(len(json_object)):
s = json_object['data'][2]
print(s)

