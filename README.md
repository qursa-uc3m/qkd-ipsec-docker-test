# QKD-IPSec Docker Testing Environment

This repository contains a testing environment for our QKD-enabled strongSwan [fork](https://github.com/qursa-uc3m/strongswan/tree/qkd) which integrates Quantum Key Distribution into (in substitution of the) the IKEv2 protocol. The setup uses Docker containers to simulate a client-server (Alice-Bob) environment for testing secure communication channels.

The testing environment is derived from the [strongX509/docker](https://github.com/strongX509/docker) project and modified to support our QKD integration testing.

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
