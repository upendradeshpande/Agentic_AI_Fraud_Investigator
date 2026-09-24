.PHONY: install run test smoke batch docker
install:
	pip install -r requirements.txt
run:
	streamlit run app/streamlit_app.py
test:
	python -m pytest -q
smoke:
	python tests/smoke_ui.py
batch:
	python -m cockpit.batch
docker:
	docker compose up --build
