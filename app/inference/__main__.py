"""Module runner for the Stream Alpha M4 inference API."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Run the inference API with uvicorn."""
    uvicorn.run(
        "app.inference.main:create_app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        factory=True,
    )


if __name__ == "__main__":
    main()
