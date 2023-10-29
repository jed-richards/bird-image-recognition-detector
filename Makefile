.PHONY: run-app
run-app:
	flask --app src/app.py run

.PHONY: tests
tests:
	python -m pytest tests/

.PHONY: test-coverage
test-coverage:
	coverage run -m pytest
	coverage report -m

.PHONY: clean
clean:
	find . -type f -name "*.pyc" | xargs rm -rf
	find . -type d -name __pycache__ | xargs rm -rf
