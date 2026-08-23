from fastapi import Header, HTTPException, status

from app.config import settings


async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    # If no key is configured (local dev), skip the check.
    if not settings.ai_service_api_key:
        return
    if x_api_key != settings.ai_service_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
