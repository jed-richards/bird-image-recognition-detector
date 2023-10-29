run-app:
	flask --app src/app.py run

tests:
	python -m pytest tests/

test-coverage:
	coverage run -m pytest
	coverage report -m

clean:
	find . -type f -name "*.pyc" | xargs rm -rf
	find . -type d -name __pycache__ | xargs rm -rf
