from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# This imports the AI logic from your analyzer.py file
from analyzer import analyze_text

MIN_WORDS = 40
MAX_WORDS = 2000
MAX_BODY_BYTES = 100_000
RATE_LIMIT = "20/hour"

app = FastAPI(title="Gotchyu")

# Note: You will need a folder named "templates" containing an "index.html" file for the home page to work
templates = Jinja2Templates(directory="templates")

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

    return analyze_text(body.text)

