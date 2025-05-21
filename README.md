# QKD-IPSec Docker Testing Environment

This repository contains a testing environment for StrongSwan with [our QKD plugins](https://github.com/qursa-uc3m/qkd-plugins-strongswan) that modify the IKEv2 protocol. The setup uses Docker containers to simulate a client-server (Alice-Bob) environment for testing secure communication channels.

The testing environment is derived from the [strongX509/docker](https://github.com/strongX509/docker) project and modified to support our QKD integration testing.

## Setting the QKD Plugin

The [qursa-uc3m/qkd-plugins-strongswan](https://github.com/qursa-uc3m/qkd-plugins-strongswan) plugins include two plugins:

- **QKD Plugin**: Implements bare QKD integration.
- **QKD-KEM Plugin**: Hybridizes QKD with Post-Quantum Cryptography using the [QKD-KEM Provider](https://github.com/qursa-uc3m/qkd-kem-provider), which depends on the [QKD-ETSI API](https://github.com/qursa-uc3m/qkd-etsi-api).

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

### Setting up QKD Certificates

To configure certificates for QKD node communication:

1. Create a `qkd_certs` folder in the root directory of the project.

2. For QuKayDee backend:

   - Download the server CA certificate from QuKayDee (named `account-<your_account_id>-server-ca-qukaydee-com.crt`)
   - Create/obtain SAE certificates:
      - `sae-1.crt` and `sae-1.key` for the master node
      - `sae-2.crt` and `sae-2.key` for the slave node
      - Place all certificates in the `qkd_certs` folder

3. Set required environment variables:

   ```bash
   export QKD_BACKEND=qukaydee
   export ACCOUNT_ID=<your_account_id>
   ```

The environment script will automatically configure paths to certificates when containers start.

The certificates are required for mutual authentication between your application and the QKD nodes. Follow the instructions in the [QuKayDee documentation](https://qukaydee.com/pages/getting_started) to generate the proper certificates and don't forget to upload your client's root certificate to their platform.

Other QKD backends like `cerberis-xgr` work analogously, although some variables like `ACCOUNT_ID` may not be necessary depending on the backend. The setup script will configure the appropriate environment variables based on the selected backend.

### Certificate Generation and Environment Preparation

Before deploying the testing environment, generate the required certificates (run outside Docker):

```bash
./scripts/gen_certs.sh
```

If you've previously run the containers, you may want to clean your Docker environment to avoid conflicts:

```bash
sudo docker system prune -a --volumes
```

### Selecting the ETSI API Version

The QKD integration supports two different ETSI API specifications:

- **ETSI 014** (default): REST-based key delivery API
- **ETSI 004**: Application Interface for traditional QKD systems

To select which API version to use:

1. Set the `ETSI_API_VERSION` environment variable:

   ```bash
   # For ETSI 004
   export ETSI_API_VERSION=004
   
   # For ETSI 014 (default)
   export ETSI_API_VERSION=014
   # Or don't set it to use the default
   ```

### Using ETSI 004 with QUBIP

For ETSI 004 testing, we use [QUBIP's ETSI-QKD-004 simulation](https://github.com/QUBIP/etsi-qkd-004/tree/ksid_sync) to provide QKD server functionality. This setup is separated into two Docker Compose files for better organization:

1. `docker-compose.yml`: Contains the StrongSwan IPSec setup
2. `qkd-etsi004.yml`: Contains the QUBIP ETSI 004 servers and key generators

### Starting the ETSI 004 Environment

To quickly start the complete ETSI 004 environment:

```bash
./scripts/start_etsi004.sh
```

This script:

- Creates the necessary ETSI 004 certificate directory
- Generates certificates for `qkd_server_alice` and `qkd_server_bob`
- Sets up the proper environment variables
- Clones the QUBIP ETSI-QKD-004 repository if not already done

### Dockerized Environment

#### Configuration Options

| Variable | Description | Default | Example Values |
|----------|-------------|---------|---------------|
| `STRONGSWAN_VERSION` | StrongSwan version | `6.0.0beta6` | `6.0.1`, `master` |
| `QKD_BACKEND` | QKD backend service | `simulated` | `qukaydee`, `cerberis-xgr`, `python_client` |
| `ACCOUNT_ID` | QuKayDee account ID | (empty) | Your provider account ID |
| `ETSI_API_VERSION` | ETSI QKD API version | `014` | `004`, `014` |
| `BUILD_QKD_ETSI` | Build QKD ETSI API | `true` | `true`, `false` |
| `BUILD_QKD_KEM` | Build QKD-KEM provider | `true` | `true`, `false` |

#### Building and Launching Containers

Build and launch containers (in this example we use QuKayDee cloud-based QKD network simulator):

```bash
ETSI_API_VERSION=014 QKD_BACKEND=qukaydee ACCOUNT_ID=<your_account_id> docker-compose -f docker-compose.yml build --no-cache && \
ETSI_API_VERSION=014 QKD_BACKEND=qukaydee ACCOUNT_ID=<your_account_id> docker-compose -f docker-compose.yml up
```

Replace `<your_account_id>` with your actual QuKayDee account ID. The environment will automatically use the certificates in your `qkd_certs` directory as specified in the configuration script.

[Configuring the QuKayDee environment](/home/javi/Documents/apps/QURSA/qkd-kem-provider/qkd_certs)

or for the ETSI 004 API:

```bash
ETSI_API_VERSION=004 QKD_BACKEND=python_client docker-compose -f docker-compose.004.yml build --no-cache && \
ETSI_API_VERSION=004 QKD_BACKEND=python_client docker-compose -f docker-compose.004.yml up
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

## Automated Testing

This repository includes a Python-based performance testing framework (`benchmark.py`) to benchmark and compare different cryptographic proposal combinations for StrongSwan's QKD integration.

The benchmark framework consists of a Python orchestrator that manages all testing aspects, a YAML configuration system for easy parameterization, direct Docker API integration for container management, and automated analysis capabilities for result processing and visualization.

### Configuration Management

The framework uses a YAML configuration file (`config/shared/benchmark_config.yml`) to control the test scenario:

```yaml
# QKD-IPSec Benchmark Configuration
docker:
  build:
    strongswan_version: "6.0.0beta6"
    build_qkd_etsi: true
    build_qkd_kem: true
    rebuild: true           # Always rebuild containers
    use_cache: false        # Don't use Docker cache

  qkd:
    etsi_api_version: "014"  # Options: "004", "014"
    backend: "qukaydee"      # Options: "simulated", "qukaydee", "cerberis-xgr", "python_client"
    account_id: "2509"       # Required for QuKayDee only

  compose_file: "docker-compose.yml"

  network:
    alice_ip: "172.30.0.3"
    bob_ip: "172.30.0.2"

test:
  network:
    apply: true
    latency: 100            # Milliseconds
    jitter: 0               # Milliseconds
    packet_loss: 5          # Percentage
    duration: "15m"         # Duration string

  iterations: 2
  
  analyze_results: true
  log_scale: true           # Use logarithmic scale for charts
```

### Network Condition Simulation

Our testing framework uses [Pumba](https://github.com/alexei-led/pumba) to simulate network conditions during tests. This allows us to evaluate QKD performance under various scenarios:

- **Latency**: Adds delay to network packets (measured in milliseconds)
- **Jitter**: Variability in packet latency (measured in milliseconds)
- **Packet Loss**: Percentage of packets that will be dropped

**These network conditions are applied to the communication between Alice and Bob during the tests, but not to their respective communication with the QKD nodes**. This allows us to isolate the performance of the QKD integration from the network conditions.

These conditions can be adjusted using command-line parameters when running the test script `run_tests.sh`. The parameters are:

| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|---------------|
| `--latency` | Network delay in milliseconds | 100 | 0, 50, 100, 200 |
| `--jitter` | Latency variation in milliseconds | 0 | 0, 5, 10, 20 |
| `--packet-loss` | Percentage of dropped packets | 5 | 0, 1, 5, 10 |
| `--no-network-conditions` | Disable network condition simulation | | |

### Running the Benchmarks

To execute the benchmark test suite with default configuration (recommended) just run the `benchmark.py` script:

```bash
docker system prune -a --volumes
python benchmark.py
```

If you want to run the tests with custom parameters, you can specify them directly in the command line. For example:

```bash
python benchmark.py --iterations 20 --latency 100 --packet-loss 5
```

Environment variables are handled automatically through the configuration file. To override the QKD backend (Edit `config/shared/benchmark_config.yml` or use command-line options):

```bash
python benchmark.py --config custom_config.yml
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

```
results/
├── <etsi_api_version>/                    # e.g., 014 or 004
│   ├── <qkd_backend>/                     # e.g., qukaydee, cerberis-xgr
│   │   ├── lat<latency>_jit<value>_loss<packet-loss>_iter<iterations>_time<timestamp>/
│   │   │   ├── test_config.json           # Test parameters
│   │   │   ├── alice_log.txt              # Alice's log output
│   │   │   ├── bob_log.txt                # Bob's log output
│   │   │   ├── capture*.pcap             # Network captures per proposal
│   │   │   ├── latencies.csv              # Handshake latency measurements
│   │   │   ├── counters.csv               # Request/response counts
│   │   │   ├── plugin_timing_summary.csv  # Plugin timing data
│   │   │   └── report.txt                 # Summary report
```

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

2. Edit the shared configuration file:

   ```bash
   nano ./config/shared/proposals_config.yml
   ```

   This YAML file contains the cryptographic proposals to test:

   ```yaml
   # Quantum Key Distribution and Post-Quantum Cryptography Test Configuration

   # Test parameters
   test_iterations: 3  # Number of times to run each proposal test

   # Cryptographic proposals to test
   # Format: encryption-integrity-keyexchange
   proposals:
   - aes128-sha256-ecp256    # Traditional elliptic curve
   - aes128-sha256-x25519    # Modern elliptic curve
   - aes128-sha256-kyber1    # Post-quantum KEM
   - aes128-sha256-qkd       # Pure QKD
   - aes128-sha256-qkd_kyber1  # Hybrid QKD+PQC

   # ESP (Encapsulating Security Payload) proposals
   esp_proposals:
   - aes128-sha256-ecp256
   - aes128-sha256-x25519
   - aes128-sha256-kyber1
   - aes128-sha256-qkd
   - aes128-sha256-qkd_kyber1
   ```

   You can modify the proposals by adding or removing items from these lists. Both Alice and Bob will automatically use the same configuration.
   For enabling intermediate IKEv2 handshakes in Strongswan, you must use a `ke1_`, `ke2_`, etc. prefix before the desired curve/qkd/kem name. The number indicates the step order.

3. Run the tests as explained above.

### Important Notes

- The benchmark suite automatically handles source environment variables for all processes
- Testing outputs are written to the `results/` directory by Docker processes running as root
- Analysis outputs should be written to `analysis/` at the project root to avoid permission issues
- After tests complete, do not modify files in the `results/` directory from the host
- The benchmark script will ask if you want to stop containers when finished
- For reproducible tests, the framework can be configured to always rebuild containers from scratch