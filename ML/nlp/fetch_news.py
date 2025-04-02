import requests
import pandas as pd
import json

# Your Polygon.io API key
API_KEY = 'BkBAQi6j6q46125xaQkX7TjBXn3es4Fi'

def fetch_articles_polygon(ticker, start_date, end_date):
    base_url = (
        f"https://api.polygon.io/v2/reference/news?"
        f"ticker={ticker}&published_utc.gte={start_date}&published_utc.lte={end_date}"
        f"&limit=1000&apiKey={API_KEY}"
    )
    all_articles = []
    url = base_url

    while url:
        response = requests.get(url)
        data = response.json()

        # Append articles from the current page
        articles = data.get('results', [])
        all_articles.extend(articles)
        
        # Retrieve the next_url from the response
        next_url = data.get('next_url')
        if next_url:
            # If the API key isn't present in the next_url, append it
            if "apiKey=" not in next_url:
                next_url += f"&apiKey={API_KEY}"
        url = next_url  # This will be None when there are no further pages

    # Print total count of articles fetched
    
    return all_articles

def extract_text(articles):
    text_data = []
    for article in articles:
        text_data.append(article.get('title', '') + " " + article.get('description', ''))
    return text_data


def fetch_articles_newsapi(query, start_date, end_date, save_file=True, max_articles=100):
    """
    Fetches news articles for a given query within a date range using NewsAPI.org.
    
    Parameters:
    - query (str): Search keyword (e.g., "Tesla").
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - api_key (str): Your NewsAPI API key.
    - save_file (bool): Whether to save results to a JSON file (default: True).
    - max_articles (int): Number of articles to fetch per request (max: 100).
    
    Returns:
    - dict: Parsed JSON response containing news articles.
    """

    api_key = "c1795e408cdb49f6ada4b9d0bd5147f4"
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": start_date,
        "to": end_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": max_articles,
        "apiKey": api_key,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        news_data = response.json()

        if save_file:
            filename = f"{query.lower()}_news_{start_date}_{end_date}.json"
            with open(filename, "w") as file:
                json.dump(news_data, file, indent=4)
            print(f"News data saved to {filename}")

        # Print first 5 articles
        for article in news_data.get("articles", [])[:5]:
            print(f"Title: {article['title']}")
            print(f"Published At: {article['publishedAt']}")
            print(f"Source: {article['source']['name']}")
            print(f"URL: {article['url']}\n")

        return news_data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def fetch_transcripts_by_date(symbol, start_date, end_date):
    """
    Fetches earnings call transcripts for a given stock symbol within a date range.
    
    Parameters:
    - symbol (str): Stock ticker symbol (e.g., 'TSLA')
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - end_date (str): End date in 'YYYY-MM-DD' format
    - api_key (str): FinancialModelingPrep API key

    Returns:
    - None (Prints the earnings call transcripts)
    """
    api_key = "dny6suJKkqwfYTotdfIdrctVvEPrYHF5"
    url = f"https://financialmodelingprep.com/stable/earning-call-transcript?symbol={symbol}&apikey={api_key}"

    response = requests.get(url)

    if response.status_code == 200:
        transcripts = response.json()
        if not transcripts:
            print("No earnings call transcripts found for this stock.")
            return
        
        print(f"\nEarnings Call Transcripts for {symbol.upper()} from {start_date} to {end_date}:\n")
        
        for transcript in transcripts:
            transcript_date = transcript.get("date")
            if transcript_date and start_date <= transcript_date <= end_date:
                print("=" * 80)
                print(f"📅 Date: {transcript_date}")
                print(f"🎤 Title: {transcript.get('title', 'N/A')}")
                print("-" * 80)
                print(transcript.get("content", "No transcript available"))
                print("\n" + "=" * 80 + "\n")
    else:
        print(f"Error {response.status_code}: {response.text}")