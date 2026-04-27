# Deploy

Operational guide for running FairAge on the Hetzner server.

## What lives in this folder

```
deploy/
├── docker-compose.yml   — service definitions for fairage-api + fairage-demo
├── nginx.conf           — path-based reverse-proxy snippet
└── README.md            — this file
```

## Architecture recap

```
[user browser]
      │  http(s)://<server>/fairage-api/...
      │  http(s)://<server>/fairage-demo/
      ▼
[nginx on port 80]   ← only public surface
      │
      ├─→ 127.0.0.1:8003 → fairage-api container (FastAPI + ONNX Runtime)
      │                    └─ reads ../artifacts/exports (mounted read-only)
      │
      └─→ 127.0.0.1:8503 → fairage-demo container (Streamlit)
                           └─ calls fairage-api over docker network
```

Both containers bind to `127.0.0.1`, never `0.0.0.0`. The public internet only sees Nginx on port 80. This pattern was forced by the cryptominer incident on `nhs-db` (April 2026) — see the NHS reference doc for the full root cause.

## First-time deploy

### 1. Make sure the Hetzner server is patched and ready

```bash
ssh root@<hetzner-host>
sudo apt list --upgradable
sudo apt update && sudo apt upgrade -y
sudo apt autoremove -y
```

### 2. Clone the repo on the server

```bash
cd ~
git clone https://github.com/NakuSurrey/fairage.git
cd fairage
```

### 3. Pull the trained ONNX models from your laptop

```bash
# from your laptop, after Surrey HPC training has produced the checkpoints
# and you have run the export + quantize scripts locally
scp artifacts/exports/age_model_int8.onnx root@<hetzner-host>:~/fairage/artifacts/exports/
scp artifacts/exports/pad_model_int8.onnx root@<hetzner-host>:~/fairage/artifacts/exports/
scp artifacts/bias_report.json root@<hetzner-host>:~/fairage/artifacts/
```

The Docker image does not bake in the models — it mounts them at runtime. New models = `scp` and `docker compose restart`, no rebuild.

### 4. Build and start the containers

```bash
cd ~/fairage
docker compose -f deploy/docker-compose.yml up -d --build
```

`-d` runs in detached mode. `--build` rebuilds images if the Dockerfiles changed.

### 5. Verify the containers came up healthy

```bash
docker compose -f deploy/docker-compose.yml ps
```

Both services should show `Up X seconds (healthy)` after about 30 seconds. If one is `unhealthy`, check the logs:

```bash
docker compose -f deploy/docker-compose.yml logs --tail 50 fairage-api
docker compose -f deploy/docker-compose.yml logs --tail 50 fairage-demo
```

### 6. Verify the local-only port binding worked

```bash
# from the Hetzner host — should respond
curl http://127.0.0.1:8003/health
curl http://127.0.0.1:8503/_stcore/health

# from your laptop — should be CLOSED, NOT reachable
nmap <hetzner-host> -p 8003,8503
```

If either port shows `open` from outside, stop and fix the binding before continuing. That is the cryptominer attack surface.

### 7. Add the FairAge block to Nginx

Open the existing Nginx site config:

```bash
sudo nano /etc/nginx/sites-available/default
```

Inside the existing `server { ... listen 80 ... }` block, paste the contents of `deploy/nginx.conf` from this repo. Save, then:

```bash
sudo nginx -t
# expected: "nginx: the configuration file /etc/nginx/nginx.conf syntax is ok"
# expected: "nginx: configuration file /etc/nginx/nginx.conf test is successful"

sudo systemctl reload nginx
```

### 8. Smoke-test the public URL

```bash
# from your laptop or any external machine
curl http://<hetzner-host>/fairage-api/health
curl -I http://<hetzner-host>/fairage-demo/
```

API should return JSON with `"status": "ok"`. Demo should return HTTP 200.

Open `http://<hetzner-host>/fairage-demo/` in a browser. The Streamlit UI should load.

## Updating the deployment

After every code change pushed to `main`:

```bash
ssh root@<hetzner-host>
cd ~/fairage
git pull origin main
docker compose -f deploy/docker-compose.yml up -d --build
```

After a model retrain (new checkpoints):

```bash
# from laptop
scp artifacts/exports/age_model_int8.onnx root@<hetzner-host>:~/fairage/artifacts/exports/
scp artifacts/exports/pad_model_int8.onnx root@<hetzner-host>:~/fairage/artifacts/exports/

# on server — restart, no rebuild needed because models are mounted
ssh root@<hetzner-host>
cd ~/fairage
docker compose -f deploy/docker-compose.yml restart
```

## Stopping the deployment

```bash
docker compose -f deploy/docker-compose.yml down
```

Add `-v` if you want to drop volumes too (this project has no named volumes, so it is a no-op for FairAge).

## Upgrading to HTTPS later (optional)

The current Nginx config is HTTP-only. To add HTTPS:

1. Point a domain at the Hetzner IP. Free options: DuckDNS, FreeDNS. Paid (£10–15/yr): namecheap, porkbun, etc.
2. Install Certbot on the server:
   ```bash
   sudo apt install certbot python3-certbot-nginx
   ```
3. Run Certbot — it auto-edits the Nginx config to add the `listen 443 ssl` block:
   ```bash
   sudo certbot --nginx -d fairage.example.com
   ```
4. No FairAge code changes needed. The containers and the docker-compose stay identical. Nginx is the only thing that knows about TLS.

## Troubleshooting

**`fairage-demo` is unhealthy and the logs say "connection refused" to fairage-api.**

The demo starts before the API finishes warming. The `depends_on: condition: service_healthy` clause should prevent this, but if the API health check is failing, the demo never starts. Check API logs first.

**`/fairage-demo/` page loads but nothing inside it updates when you click.**

Streamlit websocket is being blocked. Check `nginx.conf` includes the `proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade;` lines. Without them the websocket connection drops at the proxy.

**Public URL returns 502 Bad Gateway.**

The container is not running, or Nginx cannot reach 127.0.0.1:8003 / 8503. Run `docker compose ps` and `curl http://127.0.0.1:8003/health` from the host. The 502 means Nginx is up but the upstream is dead.

**Container logs show `model files not loaded`.**

The ONNX files are not in `~/fairage/artifacts/exports/` on the server. Re-run the `scp` step. Then `docker compose restart fairage-api`.

## Security review (run monthly)

Carry forward the NHS-incident hardening:

```bash
# 1. confirm port binding is still 127.0.0.1 only
docker port fairage-api
docker port fairage-demo

# 2. check container resource use — anything > 200% CPU sustained is a red flag
docker stats --no-stream

# 3. tail logs for anything suspicious — base64 encoded blobs, /tmp/ binary names
docker compose -f deploy/docker-compose.yml logs --tail 500 | grep -iE 'base64|/tmp/[a-z]+|wget|curl http'

# 4. update images monthly
docker compose -f deploy/docker-compose.yml pull
docker compose -f deploy/docker-compose.yml up -d --build
```
