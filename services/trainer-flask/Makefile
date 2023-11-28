IMAGE:=lrnstak-trainer
CONTAINER:=lrnstak-trainer
PORT:=5002

build:
	docker build -t $(IMAGE) .

run:
	@docker run --rm --name $(CONTAINER) -p$(PORT):5000 $(IMAGE)

start: build
	@docker stop $(CONTAINER) || echo starting...
	@docker run --rm -d --name $(CONTAINER) -p$(PORT):5000 $(IMAGE)
