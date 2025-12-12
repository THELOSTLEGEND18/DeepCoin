# DeepCoin - Cryptocurrency Sentiment Analysis

DeepCoin is an AI-driven platform designed to provide you with actionable insights into the cryptocurrency market. By leveraging advanced Natural Language Processing (NLP) and Machine Learning (ML) techniques, DeepCoin analyzes vast amounts of data from diverse sources to help you understand market sentiment, trends, and key technical indicators.

## Setup Instructions

### Prerequisites
- Docker and Docker Compose installed
- Git installed
- (Optional) Git LFS for large model files

### Quick Start

1.  **Clone Repository**:
    ```bash
    git clone https://github.com/THELOSTLEGEND18/DeepCoin.git
    cd DeepCoin
    ```

2.  **Configure Environment**:
    - Create a `.env` file in the `Backend` directory:
    ```bash
    cp Backend/.env.example Backend/.env
    ```
    - Edit `Backend/.env` and add your API keys:
      - CoinMarketCap API key
      - YouTube Data API key
      - Reddit API credentials
      - Google News SERP API key

3.  **Build and Run with Docker**:
    ```bash
    # Build both frontend and backend containers
    docker compose build
    
    # Start all services in detached mode
    docker compose up -d
    ```

4.  **Access the Application**:
    - Frontend: Open browser to `http://localhost:3000`
    - Backend API: `http://localhost:8000`
    - API Documentation: `http://localhost:8000/docs`

### Manual Build (Without Docker)

**Backend:**
```bash
cd Backend
pip install -r requirements.txt
python main.py
```

**Frontend:**
```bash
cd Frontend
npm install
npm run build
npm start
```

### Available Commands

- `docker compose up -d` - Start all services
- `docker compose down` - Stop all services
- `docker compose build frontend` - Rebuild frontend only
- `docker compose build backend` - Rebuild backend only
- `docker compose logs -f` - View logs
- `docker compose restart` - Restart all services

## Key Features

* **Comprehensive Dashboard**: A user-friendly interface providing a summary of the cryptocurrency market.

    * [![Home Page](./assets/homepage.jpeg)](./assets/homepage.jpeg)
* **Sentiment Analysis**: Understand the public mood towards cryptocurrencies by scraping data from:
    * **Reddit**: Gauge community sentiment from posts and comments using a customized FinBERT models.

        [![Reddit Sentiment](./assets/reddit.jpeg)](./assets/reddit.jpeg)
    * **YouTube**: Extract user retention from video transcripts.

        [![YouTube Sentiment](./assets/youtube.jpeg)](./assets/youtube.jpeg)
    * **News Articles**: Summarize and analyze news articles.

        [![News Sentiment](./assets/articles.jpeg)](./assets/articles.jpeg)
* **Technical Indicators**: Access crucial technical analysis tools based on historical market data, including SMA, EMA, RSI, MACD, and OBV.

    * [![Technical Indicators](./assets/indicators.jpeg)](./assets/indicators.jpeg)
* **Price Prediction**: Utilize machine learning models to forecast potential future price movements.

    * [![Price Prediction](./assets/prediction.jpeg)](./assets/prediction.jpeg)
* **Data Sources**: Aggregates information from platforms like Reddit, YouTube, Google News, and CoinMarketCap.

## Technologies Used

* **Frontend**: Next.js, React, Tailwind CSS
* **Backend**: Flask, PyTorch, Transformers
* **APIs**: CoinMarketCap, Google News SERP API, YouTube Data API, Reddit API
* **Machine Learning**: Sentiment analysis using FinBERT, price prediction using LSTM models

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
