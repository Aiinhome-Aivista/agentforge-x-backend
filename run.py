import os
from app import create_app
from flask_cors import CORS

app = create_app()
CORS(app)
if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 3006))
    debug = os.getenv("FLASK_ENV", "production").lower() == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
