#!/usr/bin/env bash
set -e

echo "========================================="
echo "  Model Forge AWS EC2 Automated Deploy"
echo "========================================="

# 1. Setup 3GB Swapfile to prevent OOM during ML training
echo "--> Configuring 3GB swap space..."
if [ ! -f /swapfile ]; then
  sudo fallocate -l 3G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=3072
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  echo "Swap created successfully."
else
  echo "Swapfile already exists."
fi
free -h

# 2. Update and install packages
echo "--> Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git build-essential libpq-dev

# 3. Pull latest code
echo "--> Preparing repository..."
cd /home/ubuntu
if [ ! -d "model-forge" ]; then
  git clone https://github.com/Abhishek-Verma0/model-forge.git
fi
cd /home/ubuntu/model-forge
git fetch --all
git checkout newbranch 2>/dev/null || git checkout main 2>/dev/null || true
git pull || true

# 4. Configure PostgreSQL
echo "--> Configuring PostgreSQL database..."
sudo -u postgres psql << 'SQL' || true
ALTER USER postgres WITH PASSWORD 'postgres';
CREATE DATABASE modelforge;
SQL

# 5. Virtual environment & ML dependencies
echo "--> Setting up Python environment..."
cd /home/ubuntu/model-forge/backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 6. Setup .env file
if [ -f /home/ubuntu/.env.production ]; then
  echo "--> Applying production .env configuration..."
  cp /home/ubuntu/.env.production .env
elif [ ! -f .env ]; then
  echo "--> Initializing default .env file..."
  cp .env.example .env
fi

# 6. Setup Systemd Service
echo "--> Registering systemd service..."
cat << 'EOF' | sudo tee /etc/systemd/system/modelforge.service
[Unit]
Description=Model Forge FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/model-forge/backend
ExecStart=/home/ubuntu/model-forge/backend/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable modelforge
sudo systemctl restart modelforge

echo "========================================="
echo "  Model Forge is LIVE! Checking status:  "
echo "========================================="
sudo systemctl status modelforge --no-pager
