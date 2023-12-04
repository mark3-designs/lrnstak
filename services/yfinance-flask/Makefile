IMAGE:=lrnstak-yfinance
CONTAINER:=lrnstak-yfinance
PORT:=5004


setup:
	@mkdir -p models && sudo chown :1000 models && chmod 775 models

build:
	docker build -t $(IMAGE) .

run:
	@docker run --rm --name $(CONTAINER) \
	    -v $(PWD)/models:/app/models \
	    -p$(PORT):5000 \
	    $(IMAGE)

start: build
	@docker stop $(CONTAINER) || echo starting...
	@docker run --rm -d --name $(CONTAINER) \
	    -v $(PWD)/models:/app/models \
	    -p$(PORT):5000 \
	    $(IMAGE)
