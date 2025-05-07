# QKD-IPSec Docker Testing Environment

> **IMPORTANT NOTE:** The [qursa-uc3m strongSwan](https://github.com/qursa-uc3m/strongswan) repository (<https://github.com/qursa-uc3m/strongswan>) is temporarily unavailable. Since this repository is required during the Docker build process, the environment cannot be built until the repository is restored. We expect it to be available again soon. Please check back later.

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
QKD_BACKEND=qukaydee ACCOUNT_ID=2509 docker-compose -f docker-compose.yml build --no-cache && QKD_BACKEND=qukaydee ACCOUNT_ID=2509 docker-compose -f docker-compose.yml up
```

## Running Tests

Start Bob (server):

```bash
docker exec -ti bob /bin/bash
export IS_TLS_SERVER=1
source /set_env.sh
./charon
```

Start Alice (client):

```bash
docker exec -ti alice /bin/bash
source /set_env.sh
./charon
```

Initiate test connection:

```bash
docker exec -ti alice /bin/bash
source /set_env.sh
swanctl --initiate --child net
```

If you run Wireshark before initiating the connection and filter for IKEv2 traffic with the filter `udp.port==500 || udp.port==4500` you should see the IKEv2 exchange.

## Testing Suite

This repository includes an automated performance testing framework to benchmark and compare different cryptographic proposal combinations for StrongSwan's QKD integration. We have implemented the following components:

- **Test Scripts**: Python scripts for Alice and Bob that automate multiple handshake iterations
- **Orchestration**: Shell script to coordinate test execution
- **Analysis**: Python script to process results and generate visualizations

### Running the Benchmark Suite

To execute the complete benchmark test suite for (e.g. the QuKayDee service):

```bash
# 1. Set required environment variables
export QKD_BACKEND=qukaydee
export ACCOUNT_ID=2507

# 2. Build and start containers (if not already running)
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d

# 3. Run the performance tests
./run_tests.sh

# 4. Analyze the results
python3 analyze_results.py
```

### What Gets Measured

The test suite measures:

- IKEv2 handshake latency with QKD enhancement
- Performance across different cryptographic proposals
- Statistical metrics (mean, standard deviation, min, max)

### Test Results

Test results are stored in two locations:

- **Raw data**: Generated in the `results/` directory
- **Analysis output**: Visualizations and reports in the `analysis/` directory

The analysis includes:

- Comparative bar charts of average handshake latencies
- Box plots showing the distribution of latencies
- Statistical summary in CSV and text formats

### Modifying the Test Suite

To test different cryptographic proposals:

1. Stop the running containers:

   ```bash
   docker-compose -f docker-compose.yml down
   ```

2. Modify the `proposals` and `esp_proposals` lists in both test scripts (`alice_tests.py` and `bob_tests.py`):

   ```python
   proposals = [
      "aes128-sha256-x25519",
      "aes128-sha256-x448",
      # Add new proposals here
   ]
   ```

   For enabling intermediate IKEv2 handshakes in Strongswan, you must use a ke1_, ke2_, etc, prefix before the desired curve/qkd/kem name. The number indicates the step order.

3. Restart the containers:

   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

4. Run the tests again:

   ```bash
   ./run_tests.sh
   ```

This approach ensures a clean environment for each test run and prevents any state from previous tests from affecting new results.

### Important Notes

- The benchmark suite automatically handles source environment variables for all processes
- Testing outputs are written to the `results/` directory by Docker processes running as root
- Analysis outputs should be written to `analysis/` at the project root to avoid permission issues
- After tests complete, do not modify files in the `results/` directory from the host