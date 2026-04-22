# Retail Customer Feedback Analyzer

Zero-cost LangGraph agent for analyzing retail reviews using free LLM APIs (Groq 14K/day or GLM-4.7-Flash). Extracts sentiment, topics, and actionable insights with full local control – no paid subscriptions required.

## Features

- Sentiment analysis (positive/negative/neutral) with confidence scores
- Key phrase extraction from customer reviews
- Automated summary generation with actionable recommendations
- Support for multiple free LLM providers (Groq, Zhipu)
- Built-in rate limiting and retry logic for API stability
- Modular agent-based architecture using LangGraph

## Architecture

The system uses a three-stage pipeline:

1. Load Reviews: Sample review loader with realistic retail data across categories (electronics, clothing, home, beauty)
2. Sentiment Analysis: Individual review processing with sentiment classification
3. Summary Generation: Aggregated insights with topic extraction and recommendations

## Prerequisites

- Python 3.10.11
- Virtual environment (recommended)
- API key from either Groq or Zhipu (free tier)

## Installation

Clone the repository and set up the environment:

```bash
git clone <repository-url>
cd retail-customer-feedback-analyzer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create a .env file in the project root:

```bash
# Provider selection (groq or zhipu)
ACTIVE_PROVIDER=zhipu

# Zhipu configuration (if using Zhipu)
ZHIPU_API_KEY=your_zhipu_api_key_here

# Groq configuration (if using Groq)
GROQ_API_KEY=your_groq_api_key_here

# Optional rate limiting (for Zhipu)
RATE_LIMIT_CALLS_PER_MINUTE=3
REQUEST_DELAY_SECONDS=2
MAX_RETRIES=5
```

## Usage

Run the main pipeline:
```bash
python main.py
```

## The system will:
1. Load sample retail reviews

2. Analyze sentiment for each review

3. Generate aggregated insights and recommendations

4. Display results in the console

## Project Structure
```bash
retail-customer-feedback-analyzer/
├── main.py                 # Workflow orchestration
├── config/
│   └── providers.py        # LLM provider configuration with rate limiting
├── agents/
│   ├── sentiment_agent.py  # Individual review analysis
│   └── summarizer_agent.py # Aggregate insights generation
├── tools/
│   └── review_loader.py    # Sample review data loader
├── requirements.txt        # Python dependencies
└── .env                    # API keys and configuration
```

## Dependencies
- langchain-openai - OpenAI-compatible LLM interface
- langchain-core - Base LangChain components
- langgraph - Workflow orchestration
- python-dotenv - Environment variable management

## Rate Limiting Notes
Zhipu API has strict rate limits (approximately 3 calls per minute). The system includes:

- Automatic rate limiting decorator

- Exponential backoff retry logic

- Batch delays between requests

- Graceful fallback when limits are exceeded

For higher throughput, use Groq which offers 14,400 requests per day on free tier.

## Testing Provider Connection

Test your API connection before running the full pipeline:

```bash
python test_zhipu.py  # For Zhipu
# or create test_groq.py for Groq
```

## Sample Output

The system produces:

- Individual review analysis with sentiment, confidence, and key phrases

- Sentiment distribution across all reviews

- Top positive and negative topics

- Recommended actions based on customer feedback

## License

This project is for educational and personal use. Review individual provider terms for API usage.