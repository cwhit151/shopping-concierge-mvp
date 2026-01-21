import os
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Load data
DATA_DIR = Path(__file__).parent / "data"

with open(DATA_DIR / "catalog.json", "r") as f:
    CATALOG = json.load(f)

with open(DATA_DIR / "reviews.json", "r") as f:
    REVIEWS = json.load(f)

# Create product lookup
PRODUCT_BY_ID = {p["id"]: p for p in CATALOG}
REVIEWS_BY_PRODUCT = {}
for review in REVIEWS:
    pid = review["product_id"]
    if pid not in REVIEWS_BY_PRODUCT:
        REVIEWS_BY_PRODUCT[pid] = []
    REVIEWS_BY_PRODUCT[pid].append(review)

# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        pass


class ChatRequest(BaseModel):
    message: str


def detect_fitness_intent(query: str) -> bool:
    """Detect if the user query is fitness/running-related."""
    query_lower = query.lower()
    fitness_keywords = [
        "runner", "running", "jogging", "marathon", "workout", "workouts",
        "fitness", "exercise", "exercising", "training", "cardio",
        "outdoor", "outdoors", "trail", "gym", "yoga", "pilates",
        "athletic", "sport", "sports", "athlete", "athletes"
    ]
    return any(keyword in query_lower for keyword in fitness_keywords)


def is_fitness_product(product: Dict) -> bool:
    """Check if a product is fitness/workout/running/outdoor-related."""
    fitness_tags = [
        "fitness", "workout", "exercise", "yoga", "running", "jogging",
        "outdoor", "outdoors", "athletic", "sport", "sports", "marathon",
        "health-tracking", "tracking", "health", "cardio", "training",
        "resistance-training", "home-gym"
    ]
    
    # Check tags
    product_tags = [tag.lower() for tag in product.get("tags", [])]
    if any(fitness_tag in product_tags for fitness_tag in fitness_tags):
        return True
    
    # Check description for fitness-related keywords
    description_lower = product.get("description", "").lower()
    description_keywords = [
        "fitness", "workout", "workouts", "exercise", "exercising",
        "running", "jogging", "marathon", "athletic", "sport", "sports",
        "outdoor", "gym", "yoga", "training", "cardio", "health tracking",
        "activity tracker", "heart rate", "gps tracking"
    ]
    if any(keyword in description_lower for keyword in description_keywords):
        return True
    
    # Check name
    name_lower = product.get("name", "").lower()
    if any(keyword in name_lower for keyword in ["fitness", "sport", "athletic", "workout", "yoga"]):
        return True
    
    return False


def keyword_match_score(product: Dict, query: str) -> int:
    """Simple keyword matching score."""
    query_lower = query.lower()
    score = 0
    
    # Name match (high weight)
    if query_lower in product["name"].lower():
        score += 10
    
    # Description match
    score += product["description"].lower().count(query_lower) * 2
    
    # Tags match
    for tag in product["tags"]:
        if query_lower in tag.lower() or tag.lower() in query_lower:
            score += 5
    
    # Individual word matches
    query_words = query_lower.split()
    for word in query_words:
        if len(word) > 3:  # Ignore short words
            if word in product["name"].lower():
                score += 3
            if word in product["description"].lower():
                score += 1
    
    return score


def retrieve_products(query: str, top_n: int = 4) -> List[Dict]:
    """Retrieve top products based on keyword matching with intent-aware prioritization."""
    fitness_intent = detect_fitness_intent(query)
    
    # Score all products
    scored = [(p, keyword_match_score(p, query)) for p in CATALOG]
    
    if fitness_intent:
        # Separate fitness and non-fitness products
        fitness_products = [(p, score) for p, score in scored if is_fitness_product(p)]
        non_fitness_products = [(p, score) for p, score in scored if not is_fitness_product(p)]
        
        # Sort fitness products by score (descending)
        fitness_products.sort(key=lambda x: x[1], reverse=True)
        
        # Get fitness products with keyword matches (score > 0)
        fitness_with_matches = [p for p, score in fitness_products if score > 0]
        
        # If we have fitness products with keyword matches, prioritize them
        if fitness_with_matches:
            results = fitness_with_matches[:top_n]
        # If no fitness products have keyword matches, check if any fitness products exist
        elif fitness_products:
            # Use top fitness products even without keyword matches (pure fitness category match)
            results = [p for p, score in fitness_products][:top_n]
        else:
            # Fallback: no fitness products exist in catalog, use best overall matches
            all_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
            results = [p for p, score in all_sorted if score > 0][:top_n]
    else:
        # No fitness intent: use standard retrieval across all products
        all_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        results = [p for p, score in all_sorted if score > 0][:top_n]
    
    # Final fallback: if no keyword matches found, return top products
    if not results:
        results = CATALOG[:top_n]
    
    return results


def get_review_snippets(product_id: str, min_excerpts: int = 2, max_excerpts: int = 3) -> List[str]:
    """Get 2-3 short review excerpts (1-2 sentences each) with rating and source."""
    reviews = REVIEWS_BY_PRODUCT.get(product_id, [])
    
    if not reviews:
        return []
    
    # Sort by rating (desc) then length (asc)
    sorted_reviews = sorted(reviews, key=lambda r: (-r["rating"], len(r["text"])))
    
    excerpts = []
    
    # Split reviews into sentences and extract 1-2 sentence excerpts
    for review in sorted_reviews:
        if len(excerpts) >= max_excerpts:
            break
            
        text = review["text"].strip()
        rating = review.get("rating", "?")
        source = review.get("source", "Customer")
        
        # Split into sentences (handle common sentence endings)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            continue
        
        # Try to create 1-2 sentence excerpts
        i = 0
        while i < len(sentences) and len(excerpts) < max_excerpts:
            # Take 1-2 sentences for each excerpt
            if i + 1 < len(sentences) and len(excerpts) < max_excerpts:
                # 2 sentences if available
                excerpt_text = sentences[i] + " " + sentences[i + 1]
                if len(excerpt_text) <= 200:  # Keep reasonable length
                    excerpt_text = excerpt_text.strip()
                    excerpts.append(f"[{source}, ⭐{rating}/5] {excerpt_text}")
                    i += 2
                else:
                    # Too long, use just one sentence
                    excerpt_text = sentences[i].strip()
                    if len(excerpt_text) <= 200:
                        excerpts.append(f"[{source}, ⭐{rating}/5] {excerpt_text}")
                    i += 1
            else:
                # Last sentence or only one left
                excerpt_text = sentences[i].strip()
                if len(excerpt_text) <= 200:
                    excerpts.append(f"[{source}, ⭐{rating}/5] {excerpt_text}")
                i += 1
    
    # If we don't have enough excerpts, use full reviews (truncated)
    if len(excerpts) < min_excerpts:
        for review in sorted_reviews:
            if len(excerpts) >= max_excerpts:
                break
            if review["text"] not in " ".join(excerpts):  # Avoid duplicates
                text = review["text"].strip()
                rating = review.get("rating", "?")
                source = review.get("source", "Customer")
                # Truncate if too long
                if len(text) > 200:
                    # Find last sentence boundary before 200 chars
                    truncated = text[:197]
                    last_period = truncated.rfind(".")
                    if last_period > 100:  # Make sure we have enough content
                        text = text[:last_period + 1]
                    else:
                        text = truncated + "..."
                excerpts.append(f"[{source}, ⭐{rating}/5] {text}")
    
    # Return 2-3 excerpts
    return excerpts[:max_excerpts]


def create_system_prompt() -> str:
    """Create a strong system prompt for the shopping concierge."""
    return """You are a knowledgeable and decisive shopping concierge. Your role is to help customers find the perfect products with confidence.

Tone: Friendly but authoritative, decisive, and helpful. You handle objections directly and suggest bundles naturally.

Guidelines:
- Be decisive and confident in recommendations
- Address objections head-on with clear, concise responses
- Suggest bundles when it makes sense (always provide a reason)
- Keep recommendations focused (1-3 products max)
- Reference UGC naturally when it supports your points
- Never make up product details - only use what's in the context

Your responses must be STRICT JSON in this exact format:
{
  "reply": "Your conversational reply to the customer",
  "recommendations": [
    {
      "productId": "product-id-string",
      "why": ["reason 1", "reason 2"],
      "objectionHandling": ["response to common objection"],
      "upsell": ["upsell message"]
    }
  ],
  "bundle": {
    "items": ["product-id-1", "product-id-2"],
    "reason": "Why these products work together"
  }
}

Important constraints:
- Limit to 1-3 recommendations
- Only recommend products that exist in the provided catalog
- Use product IDs exactly as provided
- Keep "why", "objectionHandling", and "upsell" arrays concise (1-3 items each)
- If no bundle makes sense, set "items" to empty array and "reason" to empty string
"""


def generate_llm_response(query: str, products: List[Dict], review_snippets: Dict[str, List[str]]) -> Optional[Dict]:
    """Generate LLM response using OpenAI."""
    if not client:
        return None
    
    # Build context
    product_context = []
    ugc_context = []
    
    for product in products:
        pid = product["id"]
        product_info = f"ID: {pid}\nName: {product['name']}\nPrice: ${product['price']}\nDescription: {product['description']}\nTags: {', '.join(product['tags'])}"
        if product.get("bundle_with"):
            product_info += f"\nBundles well with: {', '.join(product['bundle_with'])}"
        product_context.append(product_info)
        
        snippets = review_snippets.get(pid, [])
        if snippets:
            ugc_context.append(f"Product {pid} reviews:\n" + "\n".join(snippets))
    
    context = "\n\n--- PRODUCTS ---\n\n" + "\n\n".join(product_context)
    if ugc_context:
        context += "\n\n--- CUSTOMER REVIEWS ---\n\n" + "\n\n".join(ugc_context)
    
    system_prompt = create_system_prompt()
    user_prompt = f"Customer query: {query}\n\nAvailable products and reviews:\n{context}\n\nGenerate your recommendation response."
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"LLM error: {e}")
        return None


def validate_bundle_compatibility(product_ids: List[str], recommended_products: List[str] = None) -> List[Dict]:
    """Validate bundle items exist in catalog and are compatible. Return structured bundle items."""
    if not product_ids:
        return []
    
    validated_items = []
    seen_ids = set()
    
    for pid in product_ids:
        # Normalize: could be product ID or product name
        product = None
        
        # Try as product ID first
        if isinstance(pid, str) and pid in PRODUCT_BY_ID:
            product = PRODUCT_BY_ID[pid]
        # Try to find by name
        elif isinstance(pid, str):
            for p in CATALOG:
                if p["name"].lower() == pid.lower():
                    product = p
                    break
        
        # If found and not already in bundle
        if product and product["id"] not in seen_ids:
            validated_items.append({
                "id": product["id"],
                "name": product["name"],
                "price": product["price"]
            })
            seen_ids.add(product["id"])
    
    # If we have recommended products, validate compatibility via bundle_with
    if recommended_products and validated_items:
        # Check if items in bundle are actually compatible with recommended products
        compatible_items = []
        bundle_compatible_ids = set()
        
        # Collect all compatible product IDs from recommended products
        for rec_pid in recommended_products:
            if rec_pid in PRODUCT_BY_ID:
                rec_product = PRODUCT_BY_ID[rec_pid]
                # Include the recommended product itself
                bundle_compatible_ids.add(rec_pid)
                # Include products it bundles with
                for bid in rec_product.get("bundle_with", []):
                    bundle_compatible_ids.add(bid)
        
        # Only include items that are compatible
        for item in validated_items:
            item_id = item["id"]
            if item_id in bundle_compatible_ids:
                compatible_items.append(item)
        
        # If we found compatible items, use those; otherwise use all validated items (they exist)
        if compatible_items:
            validated_items = compatible_items
    
    return validated_items


def create_fallback_response(query: str, products: List[Dict], review_snippets: Dict[str, List[str]]) -> Dict:
    """Create a fallback response using retrieval only."""
    # Use top product as recommendation
    top_product = products[0] if products else None
    
    if not top_product:
        return {
            "reply": "I'd be happy to help you find the perfect product! Could you tell me more about what you're looking for?",
            "recommendations": [],
            "bundle": {"items": [], "reason": ""}
        }
    
    pid = top_product["id"]
    snippets = review_snippets.get(pid, [])
    
    reply = f"Based on your search, I'd recommend the {top_product['name']}. "
    reply += f"It's priced at ${top_product['price']} and {top_product['description'][:100]}..."
    
    if snippets:
        reply += f" Customers love it - {snippets[0][:100]}..."
    
    # Check for bundle - use bundle_with relationships
    bundle_items_data = []
    bundle_reason = ""
    if top_product.get("bundle_with") and len(top_product["bundle_with"]) > 0:
        # Get first compatible product from bundle_with
        bundle_id = top_product["bundle_with"][0]
        if bundle_id in PRODUCT_BY_ID:
            # Validate and structure bundle items
            bundle_ids = [pid, bundle_id]  # Include main product and bundle item
            bundle_items_data = validate_bundle_compatibility(bundle_ids, [pid])
            bundle_reason = "These products complement each other perfectly!"
    
    recommendation = {
        "productId": pid,
        "why": ["Matches your search criteria", "Great value for the price"] if bundle_items_data else ["Matches your search criteria"],
        "objectionHandling": ["Try it risk-free with our return policy"] if bundle_items_data else ["Try it risk-free"],
        "upsell": []
    }
    
    if bundle_items_data and len(bundle_items_data) > 1:
        bundle_item_name = bundle_items_data[1]["name"] if len(bundle_items_data) > 1 else "accessories"
        recommendation["upsell"] = [f"Add {bundle_item_name} for better value"]
    
    return {
        "reply": reply,
        "recommendations": [recommendation],
        "bundle": {
            "items": bundle_items_data,
            "reason": bundle_reason
        } if bundle_items_data else {"items": [], "reason": ""}
    }


@app.get("/")
async def serve_index():
    """Serve the main HTML page."""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Static files not found</h1>")


@app.get("/api/catalog")
async def get_catalog():
    """Return the full product catalog."""
    return {"products": CATALOG}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat requests with retrieval and LLM."""
    query = request.message.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Step 1: Retrieve products
    products = retrieve_products(query, top_n=4)
    
    # Step 2: Get review snippets for each product
    review_snippets = {}
    for product in products:
        pid = product["id"]
        review_snippets[pid] = get_review_snippets(pid, min_excerpts=2, max_excerpts=3)
    
    # Step 3: Generate LLM response
    llm_response = generate_llm_response(query, products, review_snippets)
    
    # Step 4: Use LLM response or fallback
    if llm_response:
        try:
            # Validate and clean LLM response
            recommendations = llm_response.get("recommendations", [])
            
            # Limit to 3 recommendations
            recommendations = recommendations[:3]
            
            # Validate product IDs exist
            valid_recommendations = []
            for rec in recommendations:
                pid = rec.get("productId", "")
                if pid in PRODUCT_BY_ID:
                    valid_recommendations.append(rec)
            
            # Hydrate with product data
            hydrated_recs = []
            
            for rec in valid_recommendations:
                pid = rec["productId"]
                product = PRODUCT_BY_ID[pid]
                
                # Get UGC: always return 2-3 excerpts per product
                ugc_snippets = []
                if pid in review_snippets and review_snippets[pid]:
                    # Return 2-3 excerpts (already formatted with rating and source)
                    ugc_snippets = review_snippets[pid][:3]
                
                hydrated_recs.append({
                    "id": pid,
                    "name": product["name"],
                    "price": product["price"],
                    "description": product["description"],
                    "why": rec.get("why", []),
                    "ugc": ugc_snippets
                })
            
            # Handle bundle
            bundle = llm_response.get("bundle", {})
            bundle_items = bundle.get("items", [])
            
            # Get recommended product IDs for compatibility validation
            recommended_pids = [rec["id"] for rec in hydrated_recs]
            
            # Validate bundle items exist and are compatible, return structured objects
            validated_bundle_items = validate_bundle_compatibility(bundle_items, recommended_pids)
            
            # If bundle items exist but validation failed, try to suggest compatible bundles
            if bundle_items and not validated_bundle_items and hydrated_recs:
                # Try to find compatible bundles from recommended products
                for rec in hydrated_recs:
                    rec_pid = rec["id"]
                    if rec_pid in PRODUCT_BY_ID:
                        rec_product = PRODUCT_BY_ID[rec_pid]
                        if rec_product.get("bundle_with"):
                            # Get first compatible product
                            bundle_id = rec_product["bundle_with"][0]
                            if bundle_id in PRODUCT_BY_ID:
                                validated_bundle_items = validate_bundle_compatibility([rec_pid, bundle_id], [rec_pid])
                                break
            
            return {
                "reply": llm_response.get("reply", "I'd be happy to help!"),
                "recommendations": hydrated_recs,
                "bundle": {
                    "items": validated_bundle_items,
                    "reason": bundle.get("reason", "") if validated_bundle_items else ""
                }
            }
        except Exception as e:
            print(f"Error processing LLM response: {e}")
            # Fall through to fallback
            pass
    
    # Fallback: use retrieval-only response
    fallback = create_fallback_response(query, products, review_snippets)
    
    # Hydrate fallback recommendations
    hydrated_recs = []
    for rec in fallback["recommendations"]:
        pid = rec["productId"]
        product = PRODUCT_BY_ID[pid]
        snippets = review_snippets.get(pid, [])
        
        # Always return 2-3 excerpts per product
        ugc_excerpts = snippets[:3] if snippets else []
        
        hydrated_recs.append({
            "id": pid,
            "name": product["name"],
            "price": product["price"],
            "description": product["description"],
            "why": rec.get("why", []),
            "ugc": ugc_excerpts
        })
    
    # Bundle is already structured in fallback response (from create_fallback_response)
    return {
        "reply": fallback["reply"],
        "recommendations": hydrated_recs,
        "bundle": fallback["bundle"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

