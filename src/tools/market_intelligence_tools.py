import requests
from typing import Optional, Dict, List

# ==================== CoinGecko API ====================
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

def get_current_coin_price(coin_id: str = "bitcoin", vs_currency: str = "usd") -> Optional[Dict]:
    """
    Fetches the current price of a cryptocurrency in a specified currency.

    Args:
        coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin").
        vs_currency (str): The fiat or crypto currency to compare against (e.g., "usd").

    Returns:
        dict: A JSON response containing the current price.
              Example: {"bitcoin": {"usd": 30000}}
        None: If the request fails.
    """
    try:
        url = f"{COINGECKO_BASE_URL}/simple/price"
        params = {"ids": coin_id, "vs_currencies": vs_currency}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching price from CoinGecko: {e}")
        return None

def get_current_coin_market_data(coin_id: str = "bitcoin") -> Optional[Dict]:
    """
    Retrieves current detailed market data for a specific cryptocurrency.

    Args:
        coin_id (str): The CoinGecko ID of the cryptocurrency.

    Returns:
        dict: Market data including 'description', 'sentiment_votes_up_percentage', 'sentiment_votes_down_percentage', 'watchlist_portfolio_users', 'market_cap_rank'.
              Example: {'description': 'This is some...' ,'sentiment_votes_up_percentage': 82.72, 'sentiment_votes_down_percentage': 17.28}

        None: If the request fails.
    """
    try:
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}"
        params = {"localization": "false", "tickers": "false", "community_data": "false", "developer_data": "false"}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        result = {}
        useful_keys = ['description', 'sentiment_votes_up_percentage', 'sentiment_votes_down_percentage', 'watchlist_portfolio_users', 'market_cap_rank']
        for key in useful_keys:
            if key == "description":
                description_text = response.json()[key].get('en', '')
                words = description_text.split()[:150]  # First 150 words
                value = ' '.join(words)
                if len(description_text.split()) > 150:
                    value += "..."
            else:
                value = response.json()[key]

            result[key] = value

        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching market data from CoinGecko: {e}")
        return None

def get_current_trending_coins() -> Optional[List[Dict]]:
    """
    Fetches the current list of trending coins on CoinGecko.

    Returns:
        list: A list of trending coins with the following info: ['name', 'id', 'market_cap_rank', 'price_btc']
        None: If the request fails.
    """
    try:
        url = f"{COINGECKO_BASE_URL}/search/trending"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response = [res['item'] for res in response.json()['coins']]
        
        result = []
        useful_keys = ['name', 'id', 'market_cap_rank', 'price_btc']
        for res in response:
            r = {}
            for key in useful_keys:
                r[key] = res[key]
            result.append(r)
        return result    

    except requests.exceptions.RequestException as e:
        print(f"Error fetching trending coins from CoinGecko: {e}")
        return None

# def get_crypto_news()

# # ==================== CoinMarketCap API ====================
# CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1"
# CMC_API_KEY = os.getenv("CMC_API_KEY", "your_coinmarketcap_api_key_here")

# def get_cmc_listings(limit: int = 100) -> Optional[Dict]:
#     """
#     Retrieves the latest cryptocurrency listings from CoinMarketCap.

#     Args:
#         limit (int): Number of cryptocurrencies to retrieve (default: 100).

#     Returns:
#         dict: Latest listings with market data.
#               Example: {"data": [{"id": 1, "name": "Bitcoin", "quote": {...}}, ...]}
#         None: If the request fails.
#     """
#     try:
#         url = f"{CMC_BASE_URL}/cryptocurrency/listings/latest"
#         headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
#         params = {"limit": limit, "convert": "USD"}
#         response = requests.get(url, headers=headers, params=params, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching listings from CoinMarketCap: {e}")
#         return None

# def get_cmc_quotes(symbol: str = "BTC") -> Optional[Dict]:
#     """
#     Gets the latest market quote for a cryptocurrency by symbol.

#     Args:
#         symbol (str): Cryptocurrency symbol (e.g., "BTC", "ETH").

#     Returns:
#         dict: Latest quote data including price, volume, market cap.
#               Example: {"data": {"BTC": {"quote": {"USD": {"price": 30000, ...}}}}}
#         None: If the request fails.
#     """
#     try:
#         url = f"{CMC_BASE_URL}/cryptocurrency/quotes/latest"
#         headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
#         params = {"symbol": symbol, "convert": "USD"}
#         response = requests.get(url, headers=headers, params=params, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching quotes from CoinMarketCap: {e}")
#         return None


# # ==================== CryptoPanic News API ====================
# CRYPTOPANIC_BASE_URL = "https://cryptopanic.com/api/v1"
# CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "your_cryptopanic_api_key_here")

# def get_crypto_news(currencies: Optional[str] = None, filter_type: str = "hot") -> Optional[Dict]:
#     """
#     Fetches the latest cryptocurrency news from CryptoPanic.

#     Args:
#         currencies (str): Comma-separated currency codes (e.g., "BTC,ETH"). None for all.
#         filter_type (str): Filter type - "rising", "hot", "bullish", "bearish", "important", "saved", "lol".

#     Returns:
#         dict: News articles with metadata.
#               Example: {"results": [{"title": "...", "url": "...", "votes": {...}}, ...]}
#         None: If the request fails.
#     """
#     try:
#         url = f"{CRYPTOPANIC_BASE_URL}/posts/"
#         params = {"auth_token": CRYPTOPANIC_API_KEY, "filter": filter_type}
#         if currencies:
#             params["currencies"] = currencies
#         response = requests.get(url, params=params, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching news from CryptoPanic: {e}")
#         return None


# # ==================== LunarCrush API ====================
# LUNARCRUSH_BASE_URL = "https://api.lunarcrush.com/v2"
# LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY", "your_lunarcrush_api_key_here")

# def get_social_metrics(symbol: str = "BTC") -> Optional[Dict]:
#     """
#     Retrieves social media metrics and sentiment for a cryptocurrency.

#     Args:
#         symbol (str): Cryptocurrency symbol (e.g., "BTC", "ETH").

#     Returns:
#         dict: Social metrics including sentiment, social volume, interactions.
#               Example: {"data": {"galaxy_score": 70, "social_volume": 5000, ...}}
#         None: If the request fails.
#     """
#     try:
#         url = f"{LUNARCRUSH_BASE_URL}/assets"
#         params = {"key": LUNARCRUSH_API_KEY, "symbol": symbol}
#         response = requests.get(url, params=params, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching social metrics from LunarCrush: {e}")
#         return None


# # ==================== Twitter API (Example - requires OAuth) ====================
# # Note: Twitter API v2 requires Bearer Token authentication
# TWITTER_API_KEY = os.getenv("TWITTER_BEARER_TOKEN", "your_twitter_bearer_token_here")

# def search_twitter_mentions(query: str = "bitcoin", max_results: int = 10) -> Optional[Dict]:
#     """
#     Searches recent tweets mentioning a specific cryptocurrency or topic.

#     Args:
#         query (str): Search query (e.g., "bitcoin", "#BTC").
#         max_results (int): Maximum number of tweets to retrieve (10-100).

#     Returns:
#         dict: Tweet data including text, metrics, author info.
#               Example: {"data": [{"id": "...", "text": "...", "public_metrics": {...}}, ...]}
#         None: If the request fails.
#     """
#     try:
#         url = "https://api.twitter.com/2/tweets/search/recent"
#         headers = {"Authorization": f"Bearer {TWITTER_API_KEY}"}
#         params = {"query": query, "max_results": max_results, "tweet.fields": "public_metrics,created_at"}
#         response = requests.get(url, headers=headers, params=params, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching Twitter data: {e}")
#         return None


# # ==================== Glassnode API ====================
# GLASSNODE_BASE_URL = "https://api.glassnode.com/v1/metrics"
# GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", "your_glassnode_api_key_here")

# def get_active_addresses(asset: str = "BTC", interval: str = "24h") -> Optional[Dict]:
#     """
#     Retrieves the number of active addresses for a blockchain.

#     Args:
#         asset (str): Asset symbol (e.g., "BTC", "ETH").
#         interval (str): Time interval - "1h", "24h", "1w", "1month".

#     Returns:
#         dict: Time-series data of active addresses.
#               Example: [{"t": 1640000000, "v": 850000}, ...]
#         None: If the request fails.
#     """
#     try:
#         url = f"{GLASSNODE_BASE_URL}/addresses/active_count"
#         params = {"a": asset, "i": interval, "api_key": GLASSNODE_API_KEY}
#         response = requests.get(url, params=params, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching active addresses from Glassnode: {e}")
#         return None

# def get_exchange_netflow(asset: str = "BTC") -> Optional[List]:
#     """
#     Gets net flow of assets into/out of exchanges (whale movements indicator).

#     Args:
#         asset (str): Asset symbol (e.g., "BTC", "ETH").

#     Returns:
#         list: Time-series of exchange net flows.
#               Example: [{"t": 1640000000, "v": -5000}, ...]
#         None: If the request fails.
#     """
#     try:
#         url = f"{GLASSNODE_BASE_URL}/transactions/transfers_volume_exchanges_net"
#         params = {"a": asset, "api_key": GLASSNODE_API_KEY}
#         response = requests.get(url, params=params, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching exchange netflow from Glassnode: {e}")
#         return None


# # ==================== Etherscan API ====================
# ETHERSCAN_BASE_URL = "https://api.etherscan.io/api"
# ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "your_etherscan_api_key_here")

# def get_eth_balance(address: str) -> Optional[Dict]:
#     """
#     Retrieves the ETH balance of a specific address.

#     Args:
#         address (str): Ethereum address (e.g., "0x742d35Cc6634C0532925a3b844Bc454e4438f44e").

#     Returns:
#         dict: Balance in Wei.
#               Example: {"status": "1", "result": "40807168564070000000000"}
#         None: If the request fails.
#     """
#     try:
#         params = {
#             "module": "account",
#             "action": "balance",
#             "address": address,
#             "tag": "latest",
#             "apikey": ETHERSCAN_API_KEY
#         }
#         response = requests.get(ETHERSCAN_BASE_URL, params=params, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching ETH balance from Etherscan: {e}")
#         return None

# def get_token_supply(contract_address: str) -> Optional[Dict]:
#     """
#     Gets the total supply of an ERC-20 token.

#     Args:
#         contract_address (str): Token contract address.

#     Returns:
#         dict: Total token supply.
#               Example: {"status": "1", "result": "21000000000000000000000000"}
#         None: If the request fails.
#     """
#     try:
#         params = {
#             "module": "stats",
#             "action": "tokensupply",
#             "contractaddress": contract_address,
#             "apikey": ETHERSCAN_API_KEY
#         }
#         response = requests.get(ETHERSCAN_BASE_URL, params=params, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching token supply from Etherscan: {e}")
#         return None


# # ==================== DeFi Llama API ====================
# DEFILLAMA_BASE_URL = "https://api.llama.fi"

# def get_protocol_tvl(protocol: str = "aave") -> Optional[Dict]:
#     """
#     Retrieves Total Value Locked (TVL) for a specific DeFi protocol.

#     Args:
#         protocol (str): Protocol name (e.g., "aave", "uniswap", "curve").

#     Returns:
#         dict: Historical TVL data.
#               Example: {"chainTvls": {...}, "tvl": [{"date": 1640000000, "totalLiquidityUSD": 15000000000}, ...]}
#         None: If the request fails.
#     """
#     try:
#         url = f"{DEFILLAMA_BASE_URL}/protocol/{protocol}"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching protocol TVL from DeFi Llama: {e}")
#         return None

# def get_all_protocols_tvl() -> Optional[List]:
#     """
#     Gets TVL data for all DeFi protocols.

#     Returns:
#         list: List of all protocols with current TVL.
#               Example: [{"name": "Aave", "tvl": 15000000000, "chain": "Ethereum", ...}, ...]
#         None: If the request fails.
#     """
#     try:
#         url = f"{DEFILLAMA_BASE_URL}/protocols"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching all protocols TVL from DeFi Llama: {e}")
#         return None


# # ==================== Token Terminal API ====================
# TOKENTERMINAL_BASE_URL = "https://api.tokenterminal.com/v2"
# TOKENTERMINAL_API_KEY = os.getenv("TOKENTERMINAL_API_KEY", "your_tokenterminal_api_key_here")

# def get_project_metrics(project_id: str = "uniswap") -> Optional[Dict]:
#     """
#     Retrieves financial metrics for a crypto project (revenue, fees, P/F ratio, etc.).

#     Args:
#         project_id (str): Project identifier (e.g., "uniswap", "aave").

#     Returns:
#         dict: Project financial metrics.
#               Example: {"project_id": "uniswap", "revenue": 500000000, "fees": 600000000, ...}
#         None: If the request fails.
#     """
#     try:
#         url = f"{TOKENTERMINAL_BASE_URL}/projects/{project_id}/metrics"
#         headers = {"Authorization": f"Bearer {TOKENTERMINAL_API_KEY}"}
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching project metrics from Token Terminal: {e}")
#         return None

# def get_market_metrics() -> Optional[List]:
#     """
#     Gets aggregated market-wide metrics across all tracked projects.

#     Returns:
#         list: Market-wide financial data.
#               Example: [{"metric": "revenue", "value": 5000000000}, ...]
#         None: If the request fails.
#     """
#     try:
#         url = f"{TOKENTERMINAL_BASE_URL}/metrics/market"
#         headers = {"Authorization": f"Bearer {TOKENTERMINAL_API_KEY}"}
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching market metrics from Token Terminal: {e}")
#         return None


# # ==================== Helper Function ====================
# def test_all_apis():
#     """
#     Tests all API functions to verify connectivity and proper configuration.
#     """
#     print("Testing Market Intelligence Agent APIs...\n")
    
#     print("1. CoinGecko - Bitcoin Price:")
#     print(get_coin_price("bitcoin", "usd"))
    
#     print("\n2. CoinMarketCap - Listings:")
#     print(get_cmc_listings(5))
    
#     print("\n3. CryptoPanic - News:")
#     print(get_crypto_news("BTC", "hot"))
    
#     print("\n4. LunarCrush - Social Metrics:")
#     print(get_social_metrics("BTC"))
    
#     print("\n5. Glassnode - Active Addresses:")
#     print(get_active_addresses("BTC"))
    
#     print("\n6. DeFi Llama - Protocol TVL:")
#     print(get_protocol_tvl("aave"))
    
#     print("\n7. Token Terminal - Project Metrics:")
#     print(get_project_metrics("uniswap"))


# if __name__ == "__main__":
#     # Example usage
#     btc_price = get_coin_price("bitcoin", "sgd")
#     if btc_price:
#         print(f"Bitcoin Price: ${btc_price['bitcoin']['sgd']:,}")
#     print(get_coin_market_data())
#     print("\n\n\n\n\n")
#     print(get_trending_coins())
#     # Uncomment to test all APIs
#     # test_all_apis()