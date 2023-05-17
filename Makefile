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

# data/processed/xy.pkl: data/raw src/process.py
# 	@echo "Processing data..."
# 	python src/process.py

# models/svc.pkl: data/processed/xy.pkl src/train_model.py
# 	@echo "Training model..."
# 	python src/train_model.py

# notebooks/results.ipynb: models/svc.pkl src/run_notebook.py
# 	@echo "Running notebook..."
# 	python src/run_notebook.py

# pipeline: data/processed/xy.pkl models/svc.pkl notebooks/results.ipynb

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache