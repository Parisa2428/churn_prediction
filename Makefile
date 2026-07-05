.PHONY: help install preprocess train api ui docker-build docker-up clean

help:  ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

preprocess:  ## Run preprocessing
	python -m src.data.preprocessing

train:  ## Train models
	python main.py

api:  ## Run FastAPI
	uvicorn src.api.main:app --reload --port 8000

ui:  ## Run Streamlit
	streamlit run src/ui/app.py

docker-build:  ## Build Docker images
	docker-compose build

docker-up:  ## Start services
	docker-compose up -d

docker-down:  ## Stop services
	docker-compose down

clean:  ## Clean temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache