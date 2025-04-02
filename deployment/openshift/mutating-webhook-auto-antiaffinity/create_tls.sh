#!/bin/bash
set -e

# Create directories if they don't exist
mkdir -p container-image

# Generate CA key and certificate
openssl genrsa -out ca.key 2048
openssl req -x509 -new -nodes -key ca.key -subj "/CN=webhook-ca" -days 365 -out ca.crt

# Generate webhook server key
openssl genrsa -out container-image/webhook.key 2048

# Create OpenSSL config file with SANs
cat <<EOF > webhook-openssl.cnf
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = antiaffinity-webhook.default.svc

[v3_req]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = antiaffinity-webhook
DNS.2 = antiaffinity-webhook.default
DNS.3 = antiaffinity-webhook.default.svc
EOF

# Generate CSR using the config file
openssl req -new -key container-image/webhook.key -out webhook.csr -config webhook-openssl.cnf

# Sign the certificate with the CA and include SANs
openssl x509 -req -in webhook.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out container-image/webhook.crt -days 365 -extensions v3_req -extfile webhook-openssl.cnf

# Clean up CSR
rm webhook.csr webhook-openssl.cnf
