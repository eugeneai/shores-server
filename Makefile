.PHONY: server celery run all

run: all

all: server

celery:
	celery -A pyramid_celery.celery_app worker --ini development.ini

server:
	pserve --reload development.ini
