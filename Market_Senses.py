import requests
import sqlite3
import time

DB_PATH = "market_qualia.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS market_leads 
                    (timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                     coin_id TEXT, price REAL, change_24h REAL)''')
    conn.commit()
    conn.close()

def fetch_market_lead():
    # Looking for the Top 5 "Lead" assets
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 5, 'page': 1}
    try:
        response = requests.get(url, params=params).json()
        conn = sqlite3.connect(DB_PATH)
        for coin in response:
            conn.execute("INSERT INTO market_leads (coin_id, price, change_24h) VALUES (?, ?, ?)",
                         (coin['id'], coin['current_price'], coin['price_change_percentage_24h']))
        conn.commit()
        conn.close()
        print(f"Market Qualia Ingested: {time.ctime()}")
    except Exception as e:
        print(f"Market Connection Severed: {e}")

if __name__ == "__main__":
    init_db()
    while True:
        fetch_market_lead()
        time.sleep(60) # Sync every minute
