import requests

url = "https://bloomberg-market-and-financial-news.p.rapidapi.com/stock/get-financials"

querystring = {"id": "aapl:us"}

headers = {
	"X-RapidAPI-Key": "40f9fd6c94msh04081f1a0f50bcap1b0be2jsn98e3b0c68bff",
	"X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())
