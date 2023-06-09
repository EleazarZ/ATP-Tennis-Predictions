install: 
	@echo "Installing..."
	python3 -m venv venv
	source venv/bin/activate
	pip install pip --upgrade
	pip install -r requirements-dev.txt
	pre-commit install
	python -m ipykernel install --user --name=atp_env
	pip install -r requirements.txt

activate:
	@echo "Activating virtual environment"
	source venv/bin/activate

setup: install activate

test:
	pytest

docs_view:
	@echo View API documentation... 
	PYTHONPATH=src pdoc src --http localhost:8080

docs_save:
	@echo Save documentation to docs... 
	PYTHONPATH=src pdoc src -o docs

process_data: data/raw src/process.py
	@echo "Processing data..."
	python src/process.py

prepare_train: data/processed/processed_data.csv src/featurize.py
	@echo "Featurizing the processing data and preparing for train..."
	python src/featurize.py

train_model: data/final/train_data.pkl src/train_model.py
	@echo "Training model..."
	python src/train_model.py

pipeline: process_data prepare_train train_model


run_service:
# docker build -t flaskapp -f app/flaskapp/Dockerfile .
# docker run -p 5000:5000 flaskapp

# docker build -t dashapp -f app/dashapp/Dockerfile .
# docker run -p 8050:8050 dashapp	

	@echo "Build or rebuild services"
	docker-compose -f app/docker-compose.yml build 

	@echo "Create and start containers"
	docker-compose -f app/docker-compose.yml up

stop_service:
	@echo "Stop and remove containers, networks"
	docker-compose -f app/docker-compose.yml down

	@echo "Removes stopped service containers"
	docker-compose -f app/docker-compose.yml rm
	

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache