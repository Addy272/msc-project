# Hosting Guide
## Easiest Demo Hosting for This Project

For this project, the easiest setup is:
1. Run the Flask app with `Waitress` on your Windows laptop
2. Open it on the same Wi-Fi/LAN, or
3. Add a temporary public URL with a tunnel if you want to share it outside your laptop

This is simpler than XAMPP for a Python/Flask app and works well for college demos.

---

## Option 1: Host on Your Laptop for a Demo

### Step 1: Install the new dependency
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 2: Start the app with Waitress
```powershell
.\host_demo.bat
```

Or run it directly:
```powershell
.\venv\Scripts\python.exe serve_waitress.py
```

### Step 3: Open it in a browser
On the same machine:
```text
http://localhost:5000
```

On another device in the same Wi-Fi network:
```text
http://YOUR_LAPTOP_IP:5000
```

To find your laptop IP on Windows:
```powershell
ipconfig
```

Look for the IPv4 address of your active Wi-Fi adapter.

---

## Option 2: Get a Public Link for College Demo

If you want a temporary public URL without deploying to a cloud host, use a tunnel.

### Cloudflare Quick Tunnel
1. Install `cloudflared`
```powershell
winget install --id Cloudflare.cloudflared
```
2. Start your app:
```powershell
.\host_demo.bat
```
3. In a second terminal, run:
```powershell
cloudflared tunnel --url http://localhost:5000
```

Cloudflare will print a public `trycloudflare.com` URL you can share for the demo.

Quick tunnels are best for demos and testing. They are not meant to be your final long-term hosting setup.

If PowerShell says `cloudflared` is not recognized right after installation, close and reopen the terminal once.
You can also run it directly with the installed path:
```powershell
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:5000
```

---

## Option 3: Host on Render

This is the best option if you want the app online without keeping your laptop on.

### Files already prepared in this project
- `render.yaml`
- `.python-version`
- `requirements.txt` with `gunicorn`

### What you need to do
1. Push this project to GitHub
2. Sign in to Render
3. Create a new Blueprint deployment from your GitHub repo
4. During setup, provide:
   - `BOOTSTRAP_ADMIN_USERNAME`
   - `BOOTSTRAP_ADMIN_PASSWORD`
5. Wait for the first deploy to finish
6. Open the Render URL and log in with that admin account

### Notes about the free plan
- Your laptop does not need to stay on
- The app will be online on a Render URL
- Free web services can spin down when idle, so the first request after inactivity may take time
- If you want a more stable always-ready demo, change the service to a paid Render plan
- Free web services use an ephemeral filesystem, so local SQLite/model changes can be lost on restart or redeploy

### Current Render start command
```text
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300 app:app
```

### Current health check path
```text
/admin/login
```

### If you later upgrade to a paid plan with a persistent disk
Attach a disk in Render and set:
```text
APP_STORAGE_ROOT=/your-disk-mount-path
```

This project is already prepared to use that path for the SQLite database, model files, and generated data.

---

## Good Defaults

- `HOST=0.0.0.0`
- `PORT=5000`
- `WAITRESS_THREADS=4`

Example custom port:
```powershell
set PORT=8000
.\host_demo.bat
```

---

## Important Notes

- This is the easiest setup for a demo, not a full production deployment.
- Keep your laptop on and connected while presenting.
- If Windows Firewall asks for permission, allow access on the network you are using.
- Because this project uses SQLite and local model files, hosting it from your own laptop is safer and simpler than many free cloud platforms.

---

## If You Want a Full Cloud Host Later

The next best option would be a Python-friendly host such as Render or PythonAnywhere, but for a quick college demo the laptop + Waitress + tunnel setup is usually much faster.
