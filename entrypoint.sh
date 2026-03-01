#!/bin/sh
# Generate config.yaml from environment variables
cat > /app/config/config.yaml <<EOF
sql_database:
  provider: sqlite
  uri: /app/data/gego.db
  database: gego

nosql_database:
  provider: mongodb
  uri: ${MONGODB_URI:-mongodb://localhost:27017}
  database: gego
EOF

echo "Config generated. MongoDB URI: ${MONGODB_URI:0:30}..."

# Start scheduler in background
/usr/local/bin/gego scheduler start --config /app/config/config.yaml &

# Start API server on Railway's PORT (default 8989)
exec /usr/local/bin/gego api --host 0.0.0.0 --port ${PORT:-8989} --config /app/config/config.yaml
