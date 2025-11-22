# Makefile
#
#   make setup      - установить зависимости
#   make run        - запустить БД и API
#   make stop       - остановить контейнеры
#   make down       - остановить и удалить контейнеры

VENV = src/venv
PIP = $(VENV)/bin/pip
UVICORN = $(VENV)/bin/uvicorn

.PHONY: setup
setup:
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install "fastapi" "uvicorn>=0.18.0" "pandas" "sqlalchemy" "psycopg2-binary" "python-multipart"

.PHONY: run
run: setup
	docker compose up -d
	sleep 5
	$(UVICORN) main:app --app-dir src --reload --host 0.0.0.0 --port 8000

.PHONY: stop
stop:
	docker compose stop

.PHONY: down
down:
	docker compose down
