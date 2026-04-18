import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI()

# ── Config ──────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PB_NEWS_URL = os.environ.get("PB_NEWS_URL", "https://pb-news.croquetwade.com")
MYCROQUET_API_URL = os.environ.get("NEWSROOM_API_URL", "https://my.croquetwade.com")
NEWSROOM_API_TOKEN = os.environ.get("NEWSROOM_API_TOKEN", "")
POLISH_MODEL = "deepseek/deepseek-v3.2"


# ── HTTP helper ──────────────────────────────────────────────────────
def _api(method: str, path: str, **kwargs):
    """Call MyCroquet /api/newsroom with subdomain bearer token."""
    url = f"{MYCROQUET_API_URL}{path}"
    headers = {"Authorization": f"Bearer {NEWSROOM_API_TOKEN}", "Content-Type": "application/json"}
    r = getattr(http_requests, method)(url, headers=headers, timeout=15, **kwargs)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r


# ── Startup check ────────────────────────────────────────────────────
@app.on_event("startup")
def _verify_api_on_startup():
    """Fail fast if NEWSROOM_API_TOKEN is missing or the API is unreachable."""
    if not NEWSROOM_API_TOKEN:
        logger.critical("FATAL: NEWSROOM_API_TOKEN is not set.")
        sys.exit(1)
    try:
        _api("get", "/api/newsroom", params={"type": "article", "scope": "all", "perPage": 1})
        logger.info("MyCroquet API ping OK at %s", MYCROQUET_API_URL)
    except Exception as exc:
        logger.critical("FATAL: MyCroquet API ping failed — check NEWSROOM_API_URL / NEWSROOM_API_TOKEN. Error: %s", exc)
        sys.exit(1)


# ── Pydantic models ────────────────────────────────────────────────
class PatchBody(BaseModel):
    title: str | None = None
    body: str | None = None
    excerpt: str | None = None
    category: str | None = None


class RejectBody(BaseModel):
    feedback: str = ""


class CreateBody(BaseModel):
    title: str
    body: str = ""
    excerpt: str = ""
    category: str = "News"


# ── Routes ──────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    """Health check: pings the MyCroquet API.

    Returns 200 with api:ok on success, 503 on error.
    Used by Docker HEALTHCHECK and Coolify monitoring.
    """
    try:
        _api("get", "/api/newsroom", params={"type": "article", "scope": "all", "perPage": 1})
        return {"status": "ok", "api": "ok", "api_url": MYCROQUET_API_URL}
    except Exception as exc:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "api": "failed", "error": str(exc)},
        )


@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/articles/{article_id}/cover")
async def cover_image(
    article_id: str,
    thumb: str = "200x140",
    filename: str | None = None,
    collection_id: str | None = None,
):
    """Redirect browser directly to PocketBase file URL (files are public — no proxy needed)."""
    if not filename or not collection_id:
        article = _api("get", f"/api/newsroom/{article_id}").json()
        filename = article.get("cover_image")
        collection_id = article.get("collectionId", "")
    if not filename:
        raise HTTPException(status_code=404, detail="No cover image")
    pb_url = (
        f"{PB_NEWS_URL}/api/files/{collection_id}/{article_id}/{filename}"
        f"?thumb={thumb}"
    )
    return RedirectResponse(url=pb_url, status_code=302)


@app.get("/api/articles")
async def list_articles(
    status: str | None = None,
    search: str | None = None,
    page: int = 1,
    perPage: int = 20,
):
    params: dict = {"type": "article", "scope": "all", "page": page, "perPage": perPage}
    if status:
        params["status"] = status
    if search:
        params["search"] = search
    r = _api("get", "/api/newsroom", params=params)
    data = r.json()
    data["totalItems"] = data.pop("total", 0)
    data["totalPages"] = -(-data["totalItems"] // perPage)  # ceiling division
    return data


@app.get("/api/articles/{article_id}")
async def get_article(article_id: str):
    r = _api("get", f"/api/newsroom/{article_id}")
    return r.json()


@app.patch("/api/articles/{article_id}")
async def patch_article(article_id: str, body: PatchBody):
    data = body.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(status_code=422, detail="No fields to update")
    r = _api("patch", f"/api/newsroom/{article_id}", json=data)
    return r.json()


@app.post("/api/articles/{article_id}/publish")
async def publish_article(article_id: str):
    r = _api("post", f"/api/newsroom/{article_id}/approve", json={})
    return r.json()


@app.post("/api/articles/{article_id}/reject")
async def reject_article(article_id: str, body: RejectBody):
    r = _api("post", f"/api/newsroom/{article_id}/reject", json={"feedback": body.feedback})
    return r.json()


@app.post("/api/articles/{article_id}/archive")
async def archive_article(article_id: str):
    r = _api("post", f"/api/newsroom/{article_id}/archive", json={})
    return r.json()


class PolishBody(BaseModel):
    body: str = ""


@app.post("/api/articles/{article_id}/polish")
async def polish_article(article_id: str, payload: PolishBody | None = None):
    # Use body from request if provided, otherwise fetch from API
    if payload and payload.body.strip():
        article_body = payload.body
    else:
        article = _api("get", f"/api/newsroom/{article_id}").json()
        article_body = article.get("body", "")
    if not article_body.strip():
        raise HTTPException(status_code=422, detail="Article body is empty")

    res = http_requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": POLISH_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a news editor for Australian croquet. "
                        "Clean up this article body (HTML). "
                        "Fix grammar, punctuation, and awkward phrasing. Trim filler words. "
                        "Keep the meaning, tone, and all facts exactly as submitted. "
                        "Also generate a short punchy headline (8 words or fewer) and a "
                        "1-2 sentence excerpt/summary. "
                        'Return JSON only: {"title": "...", "excerpt": "...", "body": "...cleaned HTML..."}'
                    ),
                },
                {"role": "user", "content": article_body},
            ],
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    data = res.json()
    if "choices" not in data:
        raise HTTPException(status_code=502, detail=f"AI polish failed: {data.get('error', {}).get('message', 'unknown error')}")
    raw = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(raw)
        return {
            "polished": parsed.get("body", raw),
            "title": parsed.get("title"),
            "excerpt": parsed.get("excerpt"),
        }
    except (json.JSONDecodeError, KeyError):
        return {"polished": raw}


@app.get("/api/stats")
async def get_stats():
    counts = {}
    for status in ("submitted", "draft", "published"):
        r = _api(
            "get",
            "/api/newsroom",
            params={"type": "article", "scope": "all", "status": status, "perPage": 1},
        )
        counts[status] = r.json().get("total", 0)
    return counts


@app.post("/api/articles")
async def create_article(body: CreateBody):
    r = _api(
        "post",
        "/api/newsroom",
        json={
            "type": "article",
            "title": body.title,
            "body": body.body,
            "excerpt": body.excerpt,
            "category": body.category,
        },
    )
    return r.json()


@app.post("/api/articles/{article_id}/accept")
async def accept_article(article_id: str):
    r = _api("post", f"/api/newsroom/{article_id}/reject", json={"feedback": ""})
    return r.json()
