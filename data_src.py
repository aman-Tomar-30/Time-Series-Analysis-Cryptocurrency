# run this file to fetch the data from coingecko and save it in the data folder as csv files.
import requests
import pandas as pd
import os
import time 

currencies = ["aave", "bitcoin", "ethereum", "binancecoin", "ripple", "cosmos", "eos", "filecoin", "maker", "tezos", "tron", "monero", "solana", "tether", "vechain"]

for currency in currencies:
    url = f"https://api.coingecko.com/api/v3/coins/{currency}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "365"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching data for {currency}: ", response.status_code)
        print(response.text)
        continue

    data = response.json()

    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    print(f"{currency} data fetched")

    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/{currency}_prices.csv", index=False)
    
    time.sleep(15)  # To respect API rate limits