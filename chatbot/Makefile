.PHONY: install embed run

# Project root, two levels up from this Makefile (chatbot/Makefile → planescapeRAG/ → repo root)
ROOT := $(shell cd ../.. && pwd)

install:
	pip install -r ../requirements.txt

# Build / refresh the embedding index over graph/entities/.
embed:
	cd $(ROOT) && python -m planescapeRAG.chatbot.embed_entities

# Start the FastAPI server. Serves both the HTML UI (at /) and the JSON API.
# Visit http://localhost:8000/ for the chat UI, or /docs for the Swagger UI.
# Binds to 0.0.0.0 so it's reachable from a remote browser (Tailscale, SSH-tunnel, LAN).
run:
	cd $(ROOT) && uvicorn planescapeRAG.chatbot.api:app --reload --host 0.0.0.0 --port 8000
