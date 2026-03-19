import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 3006))
    debug = os.getenv("FLASK_ENV", "production").lower() == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
