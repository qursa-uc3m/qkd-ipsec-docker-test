#!/bin/bash

mkdir -p ./temp_certs
cd ./temp_certs

for dir in ../bob ../alice; do
    mkdir -p $dir/pkcs8
    mkdir -p $dir/x509
    mkdir -p $dir/x509ca
done

# 1. Generate CA key and certificate
openssl genrsa -out caKey.pem 4096
openssl req -x509 -new -nodes -key caKey.pem -sha256 -days 3652 -out caCert.pem \
    -subj "/C=CH/O=Cyber/CN=Cyber Root CA"

# 2. Generate bob (server) credentials
openssl genrsa -out bobKey.pem 2048
openssl pkcs8 -topk8 -nocrypt -in bobKey.pem -out ../bob/pkcs8/bobKey.pem
openssl req -new -key bobKey.pem -out bob.csr \
    -subj "/C=CH/O=Cyber/CN=bob.strongswan.org"
openssl x509 -req -in bob.csr -CA caCert.pem -CAkey caKey.pem \
    -CAcreateserial -out ../bob/x509/bobCert.pem -days 1461 \
    -extfile <(printf "subjectAltName=DNS:bob.strongswan.org")

# 3. Generate alice (client) credentials
openssl genrsa -out aliceKey.pem 2048
openssl pkcs8 -topk8 -nocrypt -in aliceKey.pem -out ../alice/pkcs8/aliceKey.pem
openssl req -new -key aliceKey.pem -out alice.csr \
    -subj "/C=CH/O=Cyber/CN=alice@strongswan.org"
openssl x509 -req -in alice.csr -CA caCert.pem -CAkey caKey.pem \
    -CAcreateserial -out ../alice/x509/aliceCert.pem -days 1461 \
    -extfile <(printf "subjectAltName=email:alice@strongswan.org")

# 4. Copy CA certificate to both peers
cp caCert.pem ../bob/x509ca/
cp caCert.pem ../alice/x509ca/

# 5. Set proper permissions
chmod 644 ../bob/x509/*.pem ../alice/x509/*.pem ../bob/x509ca/*.pem ../alice/x509ca/*.pem
chmod 600 ../bob/pkcs8/*.pem ../alice/pkcs8/*.pem

# 6. Clean up ALL temporary files
cd ..
rm -rf ./temp_certs

echo "Certificate generation completed successfully!"