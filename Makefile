run-app:
	flask --app src/app.py run

tests:
	python -m pytest tests/
