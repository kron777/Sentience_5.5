import requests
import sqlite3
import time

# YOUR UPHOLD ACCESS TOKEN
ACCESS_TOKEN = "YOUR_UPHOLD_ACCESS_TOKEN_HERE" 
DB_PATH = "market_qualia.db"

def fetch_uphold_balance():
    url = "https://api.uphold.com/v0/me/cards"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers).json()
        total_value = 0
        conn = sqlite3.connect(DB_PATH)
        
        # Log each asset found in the Treasury
        for card in response:
            if float(card['balance']) > 0:
                asset = card['currency']
                balance = card['balance']
                print(f"Treasury Asset Detected: {asset} | {balance}")
                # Store this as 'Treasury Sensation' for the Sentinel
                conn.execute("INSERT INTO market_leads (coin_id, price, change_24h) VALUES (?, ?, ?)",
                             (f"TREASURY_{asset}", balance, 0))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Treasury Link Failed: {e}")

if __name__ == "__main__":
    while True:
        fetch_uphold_balance()
        time.sleep(300) # Check treasury every 5 minutes
