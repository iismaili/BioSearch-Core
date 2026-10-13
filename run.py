import os
import sys
import uvicorn
from download_models import ensure_models

def main():
    ensure_models(os.path.join(os.path.dirname(__file__), "models"))
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n=======================================================")
    print(f" BioSearch-Core is starting on http://{host}:{port}")
    print(f" Open http://localhost:{port} in your browser.")
    print(f"=======================================================\n")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
