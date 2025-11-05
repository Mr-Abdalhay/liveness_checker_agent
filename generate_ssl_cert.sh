#!/bin/bash

# SSL Certificate Generation Script for Liveness SDK Server
# This creates a self-signed certificate valid for 365 days

echo "Generating SSL Certificate for Liveness SDK Server..."

# Create certs directory if it doesn't exist
mkdir -p certs

# Generate private key and certificate in one step
openssl req -x509 \
  -newkey rsa:2048 \
  -keyout certs/server.key \
  -out certs/server.crt \
  -days 365 \
  -nodes \
  -subj "/C=US/ST=State/L=City/O=LivenessSDK/CN=5.22.215.77"

# Also generate with extensions for multiple IPs/hostnames
cat > certs/cert.conf <<EOF
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = LivenessSDK
CN = 5.22.215.77

[v3_req]
subjectAltName = @alt_names

[alt_names]
IP.1 = 5.22.215.77
IP.2 = 127.0.0.1
DNS.1 = localhost
EOF

# Generate certificate with extensions (alternative method)
openssl req -x509 \
  -newkey rsa:2048 \
  -keyout certs/server_alt.key \
  -out certs/server_alt.crt \
  -days 365 \
  -nodes \
  -config certs/cert.conf \
  -extensions v3_req

# Create a PEM file (certificate only) for Flutter app
cp certs/server.crt certs/server.pem

echo "Certificate generated successfully!"
echo "Files created:"
echo "  - certs/server.key (Private key - keep secret!)"
echo "  - certs/server.crt (Certificate)"
echo "  - certs/server.pem (Certificate for Flutter app)"
echo "  - certs/server_alt.* (Alternative with SAN extensions)"
echo ""
echo "Copy certs/server.pem to your Flutter app's assets/certs/ directory"
echo ""
echo "Test the certificate with:"
echo "  openssl x509 -in certs/server.crt -text -noout"
