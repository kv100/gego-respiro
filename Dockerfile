# Production Dockerfile for Gego - GEO Tracker
FROM golang:1.24-alpine AS builder

RUN apk add --no-cache git ca-certificates tzdata sqlite-dev gcc musl-dev

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=1 GOOS=linux go build -a -installsuffix cgo -ldflags="-w -s" -o gego ./cmd/gego/main.go

# Stage 2: Minimal runtime
FROM alpine:latest

RUN apk --no-cache add ca-certificates tzdata

COPY --from=builder /app/gego /usr/local/bin/gego
COPY --from=builder /app/internal/db/migrations /migrations

RUN mkdir -p /app/data /app/config /app/logs

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV GEGO_CONFIG_PATH=/app/config/config.yaml
ENV GEGO_DATA_PATH=/app/data
ENV GEGO_LOG_PATH=/app/logs

EXPOSE 8989

CMD ["/bin/sh", "/app/entrypoint.sh"]
