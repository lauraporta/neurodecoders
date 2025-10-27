# MLflow Database Migration Guide

## Prerequisites

In your environment:
```bash
pip install psycopg2-binary python-dotenv
```

## PostgreSQL Setup

### macOS

```bash
# Install PostgreSQL
brew install postgresql@14

# Start PostgreSQL service
brew services start postgresql@14

# Create a database for MLflow
createdb mlflow_db

# (Optional) Create a dedicated user
createuser mlflow_user
psql -d mlflow_db -c "ALTER USER mlflow_user WITH PASSWORD 'your_password';"
psql -d mlflow_db -c "GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;"
```

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# PostgreSQL service starts automatically
# Check status: sudo systemctl status postgresql

# Switch to postgres user and create database
sudo -u postgres createdb mlflow_db

# Create a dedicated user (optional but recommended)
sudo -u postgres createuser mlflow_user

# Set password and grant privileges
sudo -u postgres psql -d mlflow_db -c "ALTER USER mlflow_user WITH PASSWORD 'your_password';"
sudo -u postgres psql -d mlflow_db -c "GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;"

# Allow password authentication (if needed)
# Edit /etc/postgresql/*/main/pg_hba.conf and ensure you have:
# local   all             mlflow_user                             md5
# Then reload: sudo systemctl reload postgresql
```

## Configuration

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your database credentials:**
   ```bash
   POSTGRES_USER=mlflow_user
   POSTGRES_PASSWORD=your_password
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=mlflow_db
   ```

   **⚠️ Important:** The `.env` file is in `.gitignore` and will not be committed to git.

## Migration

Run the migration script to copy existing experiments and runs:
```bash
python migrate.py
```

## Configuration Changes

The following files have been updated to use PostgreSQL with credentials from `.env`:

1. **`.env`** - Contains database credentials (not committed to git)
2. **`.env.example`** - Template for environment variables
3. **`.gitignore`** - Updated to exclude `.env`
4. **`config.yaml`** - MLflow configuration (no passwords)
5. **`launch_apps.sh`** - Updated to load credentials from `.env`
6. **`neurodecoders/config.py`** - Loads config and constructs tracking URI from environment
7. **`neurodecoders/mlflow_utils/utils.py`** - Updated to use config by default
8. **`migrate.py`** - Updated to use credentials from `.env`

## Using the Database

### View Experiments in Browser

```bash
# Use launch_apps.sh (automatically loads .env)
./launch_apps.sh
```

Then open: http://localhost:5002

### In Python Scripts

The tracking URI is now automatically loaded from `.env` via `config.py`. Your existing training scripts will work without changes if they use `setup_mlflow_experiment()` from `neurodecoders.mlflow_utils.utils`.

### Environment Variable (Alternative)

You can also set the tracking URI as an environment variable:
```bash
export MLFLOW_TRACKING_URI="postgresql://mlflow_user:password@localhost/mlflow_db"
```

Or source the `.env` file:
```bash
source .env
# Then construct the URI manually or use the helper functions
```

## Updating Batch Scripts (for HPC)

If you use SLURM batch scripts (`.sbatch` files), you need to update how MLflow tracking URI is set.

### Option 1: Database on Login Node (Simple, but not recommended for production)

If PostgreSQL is running on a login node accessible from compute nodes:

```bash
# Old (file system)
export MLFLOW_TRACKING_URI=/ceph/margrie/laura/neurodecoders/mlruns
mkdir -p "$MLFLOW_TRACKING_URI"

# New (database)
export MLFLOW_TRACKING_URI=postgresql://mlflow_user:password@login-node-hostname/mlflow_db
```

⚠️ **Security Note:** Avoid hardcoding passwords in batch scripts! Instead:
1. Store credentials in `~/.env` on the cluster
2. Source it in your batch script: `source ~/.env`
3. Construct the URI from environment variables

### Option 2: Use Config File (Recommended)

The better approach is to let the Python code load from `config.yaml` automatically:

```bash
# In your .sbatch file - NO NEED to set MLFLOW_TRACKING_URI
# The Python code will read from config.yaml automatically

# Just make sure config.yaml is accessible
cd /path/to/neurodecoders

# Run your training script
python neurodecoders/encoder/mlflow_training.py \
    --model-type resnet \
    ...
```

### Important Cluster Considerations

**Network Access:**
- Compute nodes must be able to reach the PostgreSQL server
- Check firewall rules and network policies
- Test connectivity: `psql -U mlflow_user -h db-hostname -d mlflow_db` from a compute node

**Database Location Options:**
1. **Shared server:** Set up PostgreSQL on a persistent server accessible from all nodes
2. **External database:** Use a managed PostgreSQL service (AWS RDS, Google Cloud SQL, etc.)
3. **File system (fallback):** Continue using file-based tracking for HPC, sync to database later

**Performance:**
- Database writes add network latency
- For high-throughput logging, consider batching metrics
- Monitor database connection pool limits

**Recommended Setup for HPC:**
```bash
# On a persistent server/VM accessible from compute nodes:
# 1. Install PostgreSQL
# 2. Configure to accept connections from compute nodes
# 3. Set up firewall rules
# 4. Update pg_hba.conf for network access:
#    host    mlflow_db    mlflow_user    10.0.0.0/8    md5

# In config.yaml on the cluster:
mlflow:
  tracking_uri: "postgresql://mlflow_user:password@db-server.cluster.domain/mlflow_db"
```

## Artifacts Storage

- **Metadata** (parameters, metrics, tags) is stored in PostgreSQL
- **Artifacts** (models, plots, files) remain in the file system by default
- You can optionally move artifacts to cloud storage (S3, Azure Blob, etc.)

## Backup

Keep the original `mlruns/` directory backed up until you verify everything migrated correctly.

## Troubleshooting

### Check database connection
```bash
# macOS (if using default postgres user)
psql -U mlflow_user -d mlflow_db -c "SELECT COUNT(*) FROM experiments;"

# Ubuntu (switch to postgres user first)
sudo -u postgres psql -d mlflow_db -c "SELECT COUNT(*) FROM experiments;"
# Or if using mlflow_user:
psql -U mlflow_user -h localhost -d mlflow_db -c "SELECT COUNT(*) FROM experiments;"
```

### View PostgreSQL logs
```bash
# macOS (Homebrew)
tail -f /opt/homebrew/var/log/postgresql@14.log

# Ubuntu
sudo tail -f /var/log/postgresql/postgresql-*-main.log
```

### Reset database (if needed)
```bash
# macOS
dropdb mlflow_db
createdb mlflow_db
psql -d mlflow_db -c "GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;"

# Ubuntu
sudo -u postgres dropdb mlflow_db
sudo -u postgres createdb mlflow_db
sudo -u postgres psql -d mlflow_db -c "GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;"

# Then re-run migration
python migrate.py
```