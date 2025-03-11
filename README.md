# QKD-IPSec Docker Testing Environment

This repository contains a testing environment for our QKD-enabled strongSwan [fork](https://github.com/qursa-uc3m/strongswan/tree/qkd) which integrates Quantum Key Distribution into (in substitution of the) the IKEv2 protocol. The setup uses Docker containers to simulate a client-server (Alice-Bob) environment for testing secure communication channels.

The testing environment is derived from the [strongX509/docker](https://github.com/strongX509/docker) project and modified to support our QKD integration testing.

## Setting the QKD Plugin

The [qursa-uc3m/strongswan](https://github.com/qursa-uc3m/strongswan/tree/qkd) strongSwan fork includes two plugins:

- **QKD-KEM Plugin**: Hybridizes QKD with Post-Quantum Cryptography using the [QKD-KEM Provider](https://github.com/qursa-uc3m/qkd-kem-provider), which depends on the [QKD-ETSI API](https://github.com/qursa-uc3m/qkd-etsi-api).
- **QKD Plugin**: Implements bare QKD integration.

To test a specific plugin:

1. Set the `BUILD_QKD_KEM` variable in `docker-compose.yml`:

   - `"true"` - Builds QKD-KEM plugin with required dependencies (qkd-etsi-api and qkd-kem-provider)
   - `"false"` - Builds only the basic QKD plugin

2. Enable the corresponding plugin in `scripts/build_strongswan.sh`:

   - For QKD plugin: `--enable-qkd`
   - For QKD-KEM plugin: `--enable-qkd-kem`

3. Copy the plugin-specific configuration files:

```bash
# From config/<plugin_name>/ to:
alice/           # Client configuration
bob/            # Server configuration
strongswan.conf  # Main strongSwan configuration
```

*Note*: The provided configuration for the QKD-KEM plugin tests the hybridization of QKD with Kyber768.

## Setup

Generate certificates (run outside Docker):

```bash
./scripts/gen_certs.sh
```

Clean Docker environment (optional):

```bash
sudo docker system prune -a --volumes
```

Build and launch containers:

```bash
docker-compose build --no-cache && docker-compose up
```

```bash
QKD_BACKEND=qukaydee ACCOUNT_ID=2509 docker-compose -f docker-compose.dev.yml build --no-cache && QKD_BACKEND=qukaydee ACCOUNT_ID=2509 docker-compose -f docker-compose.dev.yml up
```

## Running Tests

Start Bob (server):

```bash
docker exec -ti bob /bin/bash
./charon
```

Start Alice (client):

```bash
docker exec -ti alice /bin/bash
./charon
```

Initiate test connection:

```bash
docker exec -ti alice /bin/bash
swanctl --initiate --child net
```

If you run Wireshark before initiating the connection and filter for IKEv2 traffic with the filter `udp.port==500 || udp.port==4500` you should see the IKEv2 exchange.
