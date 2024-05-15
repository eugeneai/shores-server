.PHONY: server celery run all

run: all

all: server

celery:
	celery -A pyramid_celery.celery_app worker --ini development.ini

server:
	pserve --reload development.ini


query:
	curl -X POST http://localhost:6543/sa-1.0/sparql/ba95eee8-1298-11ef-ac28-704d7b84fd9f/query/ \
	--data-urlencode 'query=select * where { ?s ?p ?o }' \
	-H "Accept: text/turtle" -H 'Content-Type: application/x-www-form-urlencoded'
