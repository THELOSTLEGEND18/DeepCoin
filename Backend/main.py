import datetime as dt
import io
import json
import os
import time
from datetime import timedelta
import tensorflow as tf
import joblib
import nltk
import numpy as np
import pandas as pd
import requests
import torch
import webvtt
import yt_dlp
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from newspaper import Article, Config
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volume import OnBalanceVolumeIndicator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

# -------------------------------
# Flask App Initialization
# -------------------------------
app = Flask(__name__)
CORS(app)

model_name = "model/finbert-sentiment"
load_dotenv()
class PredictorWrapper:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.model = None

    def _load(self):
        if self.model is None:
            self.model = tf.keras.models.load_model(self.h5_path, compile=False)

    def predict(self, X, verbose=0):
        self._load()
        return self.model.predict(X, verbose=verbose)
    
# -------------------------------
# Load Models & APIs
# -------------------------------
geminiClient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load FinBERT model
try:
    if os.path.exists("model") and os.path.isdir("model") and os.path.exists("model/config.json"):
        print("Loading model from local directory...")
        tokenizer = AutoTokenizer.from_pretrained("model")
        model = AutoModelForSequenceClassification.from_pretrained("model")
    else:
        print("Local model not found. Downloading FinBERT from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading tokenizer/model: {e}")
    print("Downloading FinBERT from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

label_map = {0: "Positive", 1: "Negative"}

# Load price prediction models
try:
    predictor = PredictorWrapper("model/price_predictor.h5")
    print("Price prediction models loaded successfully")
except Exception as e:
    print(f"Warning: Could not load price prediction models: {e}")
    predictor = None


# -------------------------------
# Utility Functions
# -------------------------------
def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions).item()
    return predicted_class


# -------------------------------
# CoinGecko API Replacement
# -------------------------------
def scrape_coingecko(coin):
    try:
        print(f"Fetching {coin} data from CoinGecko...")
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        params = {"vs_currency": "usd", "days": "365", "interval": "daily"}

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        df_price = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df_mcap = pd.DataFrame(data["market_caps"], columns=["timestamp", "Market_Cap"])
        df_vol = pd.DataFrame(data["total_volumes"], columns=["timestamp", "Volume_24h"])

        df = df_price.merge(df_mcap, on="timestamp").merge(df_vol, on="timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        df["Percent_Change_1h"] = np.nan
        df["Percent_Change_24h"] = df["price"].pct_change(1) * 100
        df["Percent_Change_7d"] = df["price"].pct_change(7) * 100
        df["Percent_Change_30d"] = df["price"].pct_change(30) * 100

        sma_window = min(20, len(df) - 1)
        sma_window_large = min(50, len(df) - 1)
        rsi_window = min(14, len(df) - 1)

        df["SMA_20"] = df["price"].rolling(window=sma_window).mean()
        df["SMA_50"] = df["price"].rolling(window=sma_window_large).mean()
        df["EMA_20"] = EMAIndicator(df["price"], window=sma_window).ema_indicator()
        df["EMA_50"] = EMAIndicator(df["price"], window=sma_window_large).ema_indicator()
        df["RSI"] = RSIIndicator(df["price"], window=rsi_window).rsi()

        macd = MACD(df["price"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()

        df["Volume_24h"].fillna(0, inplace=True)
        obv = OnBalanceVolumeIndicator(close=df["price"], volume=df["Volume_24h"])
        df["OBV"] = obv.on_balance_volume()

        technical_cols = ["SMA_20", "SMA_50", "EMA_20", "EMA_50", "RSI", "MACD", "MACD_Signal", "OBV"]
        df.dropna(subset=technical_cols, how="all", inplace=True)

        df.reset_index(inplace=True)
        df["timestamp"] = df["timestamp"].astype(str)
        df = df.replace({np.nan: None})

        return json.dumps(df.to_dict(orient="records"), default=str)

    except Exception as e:
        print(f"Error fetching CoinGecko data for {coin}: {e}")
        return json.dumps({"error": str(e)})


# -------------------------------
# YouTube Scraper
# -------------------------------
def scrape_yt(url):
    url = url.strip().replace("'", "").replace('"', "").replace("\n", "")
    print(f"Scraping YouTube: {url}")

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "writeinfojson": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        output = {"Description": "", "Transcript": "", "Average Retention": -1}
        info = ydl.extract_info(url, download=False)
        output["Description"] = info.get("description", "No description available.")

        sponsor_segments = []
        try:
            response = requests.get(f"https://sponsor.ajay.app/api/skipSegments?videoID={info['id']}&category=sponsor")
            if response.status_code == 200:
                sponsor_segments = response.json()
        except Exception:
            sponsor_segments = []

        retention_data = info.get("heatmap")
        if retention_data:
            non_sponsor_retention = [
                r["value"]
                for r in retention_data
                if not any(seg["segment"][0] <= r["start_time"] <= seg["segment"][1] for seg in sponsor_segments)
            ]
            if non_sponsor_retention:
                output["Average Retention"] = round(np.mean(non_sponsor_retention), 4)

        try:
            subs = info.get("requested_subtitles", {})
            en_sub = subs.get("en") or subs.get("en-auto")
            if en_sub:
                subtitle_url = en_sub["url"]
                subtitle_content = ydl.urlopen(subtitle_url).read().decode("utf-8")
                vtt = webvtt.read_buffer(io.StringIO(subtitle_content))
                transcript_lines = [caption.text.strip() for caption in vtt]
                output["Transcript"] = " ".join(transcript_lines)
            else:
                output["Transcript"] = "No English subtitles available."
        except Exception:
            output["Transcript"] = "Error extracting transcript."

        return output


# -------------------------------
# Reddit Scraper
# -------------------------------
def scrape_reddit(subreddit):
    subreddit = subreddit.strip().replace("\n", "").replace("'", "").replace('"', "")
    print(f"Scraping Subreddit: {subreddit}")
    url = f"https://www.reddit.com/r/{subreddit}/rising/.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    output = {}

    response = requests.get(url=url, headers=headers)
    if response.status_code != 200:
        return "Could not scrape Reddit"

    data = response.json()
    for i in range(10):
        try:
            post = data["data"]["children"][i]["data"]
            output[i] = {
                "title": post["title"],
                "description": post["selftext"],
                "url": "https://www.reddit.com" + post["permalink"],
                "upvote_ratio": post["upvote_ratio"],
                "comments": {},
            }
        except:
            break

    overall_sentiment = 0
    for i in output:
        url = output[i]["url"] + ".json"
        r = requests.get(url=url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            try:
                for j in range(10):
                    sentiment = label_map[predict_sentiment(data[1]["data"]["children"][j]["data"]["body"])]
                    output[i]["comments"][j] = {
                        "text": data[1]["data"]["children"][j]["data"]["body"],
                        "sentiment": sentiment,
                        "upvotes": data[1]["data"]["children"][j]["data"]["ups"],
                    }
                    overall_sentiment += 1 if sentiment == "Positive" else -1
            except:
                pass
    output["sentiment"] = overall_sentiment
    return output


# -------------------------------
# YouTube Search
# -------------------------------
def search_youtube(search_query):
    search_query = search_query.strip().replace("\n", "").replace("'", "").replace('"', "")
    print(f"Searching YouTube: {search_query}")
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "geo_bypass": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch10:{search_query}", download=False)
            videos = result.get("entries", [])
            top_videos = {
                i: {"title": v.get("title", "Unknown"), "url": v.get("url", "")}
                for i, v in enumerate(sorted(videos, key=lambda x: x.get("view_count", 0), reverse=True)[:6])
            }
            return top_videos
    except Exception as e:
        return {"error": "Failed to search YouTube", "message": str(e)}


# -------------------------------
# News Articles via SerpAPI
# -------------------------------
def getArticles(search_query):
    serp_api_key = os.getenv('SERP_API_KEY')
    if not serp_api_key:
        print("ERROR: SERP_API_KEY not found")
        return {"error": "SERP_API_KEY not configured"}

    url = f"https://serpapi.com/search?engine=google_news&q={search_query}&api_key={serp_api_key}"
    print(f"Fetching articles for: {search_query}")

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"error": f"Could not scrape Google News: {response.status_code}"}

        data = response.json()
        articles = data.get("news_results", [])

        if not articles:
            return {"info": "No articles found"}

        results = {}
        for i, article in enumerate(articles[:5]):
            link = article.get("link")
            if not link:
                continue
            config = Config()
            config.request_timeout = 3
            try:
                news_article = Article(link, config=config)
                news_article.download()
                news_article.parse()
                news_article.nlp()
                results[i] = {
                    "title": news_article.title,
                    "link": link,
                    "text": news_article.text,
                    "summary": news_article.summary,
                    "keywords": news_article.keywords,
                }
            except Exception:
                continue

        return results
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# NEW LSTM PRICE PREDICTOR (LOG-RETURNS)
# ============================================================
def price_predictor(coin):
    if predictor is None:
        return json.dumps({"error": "Price prediction model not loaded"})

    # Fetch EXACT 365 days
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": "usd", "days": "365", "interval": "daily"}

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return json.dumps({"error": "CoinGecko fetch failed"})

    data = resp.json()
    prices = [p[1] for p in data["prices"]]
    timestamps = [p[0] for p in data["prices"]]

    if len(prices) < 365:
        return json.dumps({"error": "Not enough data"})

    dates = pd.to_datetime(timestamps, unit="ms").date

    # Log returns
    logp = np.log(np.array(prices))
    lrets = np.diff(logp)

    # Normalize returns (per coin)
    mean = lrets.mean()
    std = lrets.std() + 1e-8
    norm_lrets = (lrets - mean) / std

    X_input = norm_lrets[-365:].reshape(1, 365, 1).astype(np.float32)

    # Predict normalized future log-returns
    pred_norm = predictor.predict(X_input, verbose=0)[0]

    # De-normalize
    pred_lrets = pred_norm * std + mean

    # Reconstruct prices
    last_price = prices[-1]
    future_prices = []
    p = last_price
    for lr in pred_lrets:
        p = p * np.exp(lr)
        future_prices.append(float(p))

    future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 8)]

    hist_df = pd.DataFrame({"date": dates[-100:], "price": prices[-100:]})
    fut_df = pd.DataFrame({"date": future_dates, "price": future_prices})

    combined = pd.concat([hist_df, fut_df], ignore_index=True)
    return combined.to_json(date_format="iso", orient="records")


# -------------------------------
# Cache Dictionaries
# -------------------------------
yt_cache = {}
reddit_cache = {}
articles_cache = {}
coingecko_cache = {}
priceprediction_cache = {}
summary_cache = {}


# ============================================================
# NEW /v1/predictPrice (USING LOG-RETURNS LSTM)
# ============================================================
@app.route("/v1/predictPrice", methods=["POST"])
def predictPrice():
    coin = request.json["coin"].lower()

    if coin in priceprediction_cache:
        return priceprediction_cache[coin]

    data = price_predictor(coin)
    priceprediction_cache[coin] = data
    return data


# -------------------------------
# Flask Routes (UNCHANGED)
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the DeepCoin API (CoinGecko Edition)"})


@app.route("/v1/scrapeYoutube", methods=["POST"])
def scrapeYoutube():
    coin = request.json["coin"].lower()
    if coin in yt_cache:
        return jsonify(yt_cache[coin])
    search_query = coin + " latest news"
    videos = search_youtube(search_query)
    for k, v in videos.items():
        try:
            link = v.get("url")
            if link:
                v.update(scrape_yt(link))
        except Exception as e:
            v["scrape_error"] = str(e)
    yt_cache[coin] = videos
    return jsonify(videos)


@app.route("/v1/scrapeReddit", methods=["POST"])
def scrapeReddit():
    coin = request.json["coin"].lower()
    if coin in reddit_cache:
        return json.dumps(reddit_cache[coin])
    posts = scrape_reddit(coin)
    reddit_cache[coin] = posts
    return json.dumps(posts)


@app.route("/v1/scrapeArticles", methods=["POST"])
def scrapeArticles():
    coin = request.json["coin"].lower()
    if coin in articles_cache:
        return jsonify(articles_cache[coin])
    data = getArticles(coin)
    articles_cache[coin] = data
    return jsonify(data)


@app.route("/v1/scrapeCoinGecko", methods=["POST"])
def scrapeCoinGecko():
    coin = request.json["coin"].lower()
    if coin in coingecko_cache:
        return coingecko_cache[coin]
    data = scrape_coingecko(coin)
    coingecko_cache[coin] = data
    return data


@app.route("/v1/analyzeCoin", methods=["POST"])
def analyzeCoin():
    coin = request.json["coin"].lower()
    if coin in summary_cache:
        return jsonify(summary_cache[coin])

    try:
        coin_data = json.loads(scrape_coingecko(coin))
        last_7_days = {str(i): d for i, d in enumerate(coin_data[-7:])}
    except Exception:
        last_7_days = {}

    try:
        reddit_data = reddit_cache.get(coin, scrape_reddit(coin))
    except Exception:
        reddit_data = {}

    try:
        articles_data = articles_cache.get(coin, getArticles(coin))
    except Exception:
        articles_data = {}

    try:
        youtube_data = yt_cache.get(coin, search_youtube(coin + " latest news"))
    except Exception:
        youtube_data = {}

    try:
        price_data = json.loads(price_predictor(coin))
        next_7_days = {str(i): d for i, d in enumerate(price_data[-7:])}
    except Exception:
        next_7_days = {}

    prompt = f"""
    You are an expert crypto analyst. Analyze the given coin using the data below and summarize insights.
    Reddit: {reddit_data}
    Articles: {articles_data}
    YouTube: {youtube_data}
    Last 7 Days Market Data: {last_7_days}
    Next 7 Days Predictions: {next_7_days}
    """

    try:
        response = geminiClient.models.generate_content(
            model="gemini-flash-latest", contents=prompt
        )
        output = {"analysis": response.text, "model": "gemini-flash-latest"}
    except Exception as e:
        output = {
            "analysis": "Summary unavailable due to API error.",
            "model": "fallback",
            "error": str(e),
        }

    summary_cache[coin] = output
    return jsonify(output)


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    app.run(debug=os.getenv("DEBUG", False), host="0.0.0.0", port=8000)