from fastapi import Header, HTTPException, status

async def verify_api_token(x_api_token: str = Header(None)) -> str:
    """Header-based authentication verification dependency.
    
    Validates X-API-Token format if provided.
    """
    if x_api_token is not None and len(x_api_token) < 5:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Token format."
        )
    return x_api_token or ""
