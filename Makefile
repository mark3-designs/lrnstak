
setup:
	# allows the lrnstak-registry permissions when running via docker-compose to write to the mounted director for storing model files and metadata.
	@mkdir -p models && sudo chown :1000 models && sudo chmod 775 models

test:
	docker build --target tests -f Dockerfile.tests -t test_lrnstak .

package: test
	@mkdir -p dist/
	@rm -rf dist/* || echo OK
	rm services/*-flask/lrnstak-*.whl || echo OK
	@docker run -v $(PWD):/build --workdir /build --rm python:3.8-slim python setup.py sdist bdist_wheel

build: package
	cp dist/lrnstak-*.whl services/predictions-flask/
	cp dist/lrnstak-*.whl services/registry-flask/
	cp dist/lrnstak-*.whl services/trainer-flask/
	@docker-compose build

start:
	@docker-compose up --build -d

stop:
	@docker-compose stop || echo OK

clean: stop
	@docker-compose rm -f || echo OK


