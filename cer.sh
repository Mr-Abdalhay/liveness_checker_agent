#!/bin/bash

# SSL Certificate Generation Script for Liveness SDK Server
# This creates a self-signed certificate valid for 365 days

echo "Generating SSL Certificate for Liveness SDK Server..."

# Create certs directory if it doesn't exist
mkdir -p certs

# Generate private key and certificate
openssl req -x509 \
  -newkey rsa:4096 \
  -keyout certs/server.key \
  -out certs/server.crt \
  -days 365 \
  -nodes \
  -subj "/C=US/ST=State/L=City/O=LivenessSDK/CN=5.22.215.77" \
  -addext "subjectAltName=IP:5.22.215.77,IP:127.0.0.1,DNS:localhost"

# Create a PEM file (certificate only) for Flutter app
cp certs/server.crt certs/server.pem

echo "Certificate generated successfully!"
echo "Files created:"
echo "  - certs/server.key (Private key - keep secret!)"
echo "  - certs/server.crt (Certificate)"
echo "  - certs/server.pem (Certificate for Flutter app)"
echo ""
echo "Copy certs/server.pem to your Flutter app's assets/certs/ directory"
