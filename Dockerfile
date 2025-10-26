# Multi-stage Dockerfile
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o /gointellect ./cmd/gointellect

FROM alpine:3.18
COPY --from=builder /gointellect /usr/local/bin/gointellect
ENTRYPOINT ["/usr/local/bin/gointellect"]
