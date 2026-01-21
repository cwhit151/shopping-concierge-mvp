# AI Shopping Concierge MVP

A fully functional AI-powered shopping concierge demo built with FastAPI and OpenAI. This single-service application provides intelligent product recommendations, handles customer objections, and suggests bundles using LLM-powered retrieval over a product catalog and user-generated content.

## Features

- 🤖 **AI-Powered Recommendations**: Uses OpenAI GPT to generate personalized product recommendations
- 🔍 **Intelligent Retrieval**: Keyword-based product matching across names, descriptions, and tags
- 💬 **Conversational Interface**: Natural language chat interface for product inquiries
- 📦 **Bundle Suggestions**: Automatically suggests product bundles based on catalog relationships
- ⭐ **UGC Integration**: Incorporates user reviews and ratings to support recommendations
- 🛡️ **Graceful Fallbacks**: Works even if OpenAI API is unavailable (uses retrieval-only mode)
- 🎨 **Modern Dark UI**: Clean, responsive interface with product grid and chat panel

## Tech Stack

- **Python 3.11+**
- **FastAPI** - Modern web framework for building APIs
- **Uvicorn** - ASGI server
- **OpenAI SDK** - LLM integration for intelligent responses
- **Plain HTML/CSS/JS** - Lightweight frontend (no framework required)

## Project Structure

```
shopping-concierge-mvp/
├── app.py                 # FastAPI backend application
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment configuration
├── README.md             # This file
├── data/
│   ├── catalog.json      # Product catalog (10 products)
│   └── reviews.json      # User reviews (3-6 per product)
└── static/
    └── index.html        # Frontend UI
```

## Local Setup

### Prerequisites

- Python 3.11 or higher
- OpenAI API key (optional - app works with fallback mode if not provided)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd shopping-concierge-mvp
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"  # Optional
   export OPENAI_MODEL="gpt-4o-mini"          # Optional, defaults to gpt-4o-mini
   ```

   On Windows:
   ```cmd
   set OPENAI_API_KEY=your-api-key-here
   set OPENAI_MODEL=gpt-4o-mini
   ```

5. **Run the application:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 10000
   ```

6. **Open your browser:**
   Navigate to `http://localhost:10000`

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | No | - | Your OpenAI API key. If not provided, the app uses fallback retrieval-only mode. |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | The OpenAI model to use (e.g., `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`). |

## Render Deployment

### Step 1: Push to GitHub

1. Create a new GitHub repository
2. Push this codebase to your repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/shopping-concierge-mvp.git
   git push -u origin main
   ```

### Step 2: Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Render will auto-detect the `render.yaml` configuration
5. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `OPENAI_MODEL`: `gpt-4o-mini` (or your preferred model)
6. Click "Create Web Service"
7. Wait for deployment to complete

### Step 3: Access Your App

Your app will be available at `https://your-service-name.onrender.com`

**Note:** Render's free tier may spin down after inactivity. First request after inactivity may take 30-60 seconds.

## API Endpoints

### `GET /`

Serves the main HTML page (`static/index.html`).

### `GET /api/catalog`

Returns the complete product catalog.

**Response:**
```json
{
  "products": [
    {
      "id": "prodigy-wireless-earbuds",
      "name": "Prodigy Wireless Earbuds",
      "price": 79,
      "description": "...",
      "tags": ["audio", "wireless"],
      "bundle_with": ["tech-charge-cable"]
    }
  ]
}
```

### `POST /api/chat`

Processes a chat message and returns recommendations.

**Request:**
```json
{
  "message": "Tell me about wireless earbuds"
}
```

**Response:**
```json
{
  "reply": "Based on your search, I'd highly recommend...",
  "recommendations": [
    {
      "id": "prodigy-wireless-earbuds",
      "name": "Prodigy Wireless Earbuds",
      "price": 79,
      "description": "...",
      "why": ["Great sound quality", "Long battery life"],
      "ugc": ["[TikTok] These are actually insane..."]
    }
  ],
  "bundle": {
    "items": ["Prodigy Wireless Earbuds", "TechCharge USB-C Cable"],
    "reason": "Perfect for music lovers on the go"
  }
}
```

## Demo Script

Here's a 5-step demo script to showcase the full capabilities:

### Step 1: Product Recommendation
**User:** "Show me wireless earbuds"

**Expected:** AI recommends Prodigy Wireless Earbuds with reasons (noise cancellation, battery life) and includes positive UGC reviews.

### Step 2: Objection Handling
**User:** "But $79 seems expensive"

**Expected:** AI addresses the price concern by emphasizing value (30-hour battery, ANC quality), cites reviews mentioning "worth every penny," and may suggest bundle savings.

### Step 3: Product Details
**User:** "Tell me about the SmartWatch Sport Edition. Is it worth it?"

**Expected:** AI provides detailed recommendation with fitness features, GPS tracking, week-long battery, and includes relevant reviews.

### Step 4: Bundle Upsell
**User:** "I'm interested in the wireless mouse"

**Expected:** AI recommends the Ergo Wireless Mouse and suggests a bundle with the Slim Wireless Keyboard and Adjustable Laptop Stand, explaining the ergonomic benefits of the complete setup.

### Step 5: Alternative Search
**User:** "I need something for home workouts"

**Expected:** AI recommends fitness products (Yoga Mat, Resistance Bands) with explanations and may suggest bundling them for a complete home gym solution.

## How It Works

### Retrieval System

1. **Keyword Matching**: Extracts keywords from user query
2. **Product Scoring**: Scores products based on matches in:
   - Product name (highest weight)
   - Product tags (medium weight)
   - Product description (lower weight)
3. **Top Selection**: Selects top 4 matching products
4. **Review Retrieval**: Pulls 2-3 highest-rated, shortest review snippets per product

### LLM Generation

1. **Context Building**: Creates context with retrieved products and reviews
2. **System Prompt**: Provides strict JSON response format with schema
3. **Response Generation**: Uses OpenAI API with `response_format={"type":"json_object"}`
4. **Validation**: Validates product IDs exist, limits recommendations to 3
5. **Hydration**: Enriches response with full product data for UI

### Fallback Mode

If `OPENAI_API_KEY` is missing or API call fails:
- Uses retrieval-only recommendations
- Creates natural-language responses from product data
- Still suggests bundles based on catalog relationships
- Ensures demo always works

## Limitations & Future Enhancements

**Current Limitations:**
- Simple keyword-based retrieval (no semantic search)
- Fixed product catalog (10 products)
- Limited to 3 recommendations per query
- No conversation history/memory

**Potential Enhancements:**
- Vector embeddings for semantic search
- Conversation memory/context
- User preferences and history
- Real-time inventory integration
- A/B testing for recommendation strategies
- Analytics and recommendation performance tracking

## Troubleshooting

### App won't start locally
- Ensure Python 3.11+ is installed: `python --version`
- Check all dependencies installed: `pip install -r requirements.txt`
- Verify port 10000 is available

### OpenAI API errors
- App automatically falls back to retrieval-only mode
- Check API key is correct: `echo $OPENAI_API_KEY`
- Verify API key has credits/quota available
- Check model name is valid

### Products not showing
- Verify `data/catalog.json` exists and is valid JSON
- Check browser console for errors
- Ensure `/api/catalog` endpoint works: `curl http://localhost:10000/api/catalog`

### Render deployment fails
- Check `render.yaml` syntax is correct
- Verify environment variables are set in Render dashboard
- Check build logs in Render dashboard for specific errors
- Ensure `requirements.txt` has all dependencies

## License

This project is provided as-is for demonstration purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in `app.py`
3. Check Render deployment logs if deploying

---

**Built with ❤️ for AI-powered shopping experiences**

