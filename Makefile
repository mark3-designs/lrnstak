
setup:
	# allows the lrnstak-registry permissions when running via docker-compose to write to the mounted director for storing model files and metadata.
	@mkdir -p models && sudo chown :1000 models && chmod 775 models

build:
	@docker-compose build

start:
	@docker-compose up --build -d

stop:
	@docker-compose stop || echo OK

clean: stop
	@docker-compose rm -f || echo OK
