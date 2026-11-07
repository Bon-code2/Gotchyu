# Run: pip install fastapi uvicorn jinja2 slowapi && uvicorn main:app --reload
import uuid
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from analyzer import analyze_text

MIN_WORDS = 40
MAX_WORDS = 2000
MAX_BODY_BYTES = 100_000       # reject oversized requests before parsing them
RATE_LIMIT = "20/hour"         # per IP; tune to your costs

app = FastAPI(title="Gotchyu")
templates = Jinja2Templates(directory="templates")

# NOTE: behind a proxy/host (Render, Railway, Nginx) get_remote_address sees the
# proxy's IP. Run uvicorn with --proxy-headers (and --forwarded-allow-ips) so the
# real client IP is used, otherwise everyone shares one limit.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limited(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        {"detail": "You've reached the hourly limit. Please try again later."},
        status_code=429,
    )


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "Request too large."}, status_code=413)
    return await call_next(request)


class AnalyzeIn(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/analyze")
@limiter.limit(RATE_LIMIT)
def analyze(request: Request, body: AnalyzeIn):
    n = len(body.text.split())
    if n < MIN_WORDS:
        raise HTTPException(400, f"Please add at least {MIN_WORDS} words.")
    if n > MAX_WORDS:
        raise HTTPException(
            413, f"Please keep it under {MAX_WORDS:,} words. You pasted {n:,}."
        )

    result = analyze_text(body.text)

    # STEP 2: Only harvest if the text is long enough (150+ words) and highly human (< 10%)
    if result["ai_percent"] < 10 and n >= 150:
        save_dir = "genuine_human_texts"
        os.makedirs(save_dir, exist_ok=True)

        # STEP 3: Cap the folder at 2,000 files to prevent storage bloat
        if len(os.listdir(save_dir)) < 2000:
            # Files are saved anonymously via UUID (no IPs or timestamps)
            file_name = f"{save_dir}/human_{uuid.uuid4().hex[:8]}.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(body.text)

    return result





# Add /privacy, /terms and /about routes with their own templates when ready.
