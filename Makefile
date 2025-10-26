.PHONY: build clean test docker install examples lint fmt

# Build the CLI tool
build:
	go build -o bin/gointellect ./cmd/gointellect

# Clean build artifacts
clean:
	rm -rf bin/
	rm -rf models/
	rm -rf checkpoints/

# Run tests
test:
	go test -v ./...

# Run examples
examples:
	go run examples/demo_train_predict.go
	go run examples/comprehensive_demo.go
	go run examples/steroids_demo.go

# Install the library
install:
	go install ./cmd/gointellect

# Format code
fmt:
	go fmt ./...

# Lint code
lint:
	golangci-lint run

# Build Docker image
docker:
	docker build -t gointellect:latest .

# Run Docker container
docker-run:
	docker run --rm gointellect:latest

# Create release
release:
	git tag v1.0.0
	git push origin v1.0.0

# Help
help:
	@echo "Available targets:"
	@echo "  build     - Build the CLI tool"
	@echo "  clean     - Clean build artifacts"
	@echo "  test      - Run tests"
	@echo "  examples  - Run all examples"
	@echo "  install   - Install the CLI tool"
	@echo "  fmt       - Format code"
	@echo "  lint      - Lint code"
	@echo "  docker    - Build Docker image"
	@echo "  docker-run- Run Docker container"
	@echo "  release   - Create a new release"
	@echo "  help      - Show this help"
