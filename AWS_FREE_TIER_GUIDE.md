# AWS Free Tier Hosting Guide
## Best Fit for This Project: Elastic Beanstalk Single Instance

For this Flask project, the easiest AWS option is:
1. Build a source bundle zip
2. Deploy it to AWS Elastic Beanstalk
3. Use a single instance environment

This avoids XAMPP, avoids keeping your laptop on, and does not require AWS CLI on your machine.

---

## Files Already Prepared

- `application.py`
- `Procfile`
- `.ebextensions/01-environment.config`
- `.ebextensions/02-single-instance.config`
- `build_aws_bundle.ps1`

---

## Step 1: Build the AWS Upload Bundle

From PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_aws_bundle.ps1
```

This creates:

```text
dist\aws-elastic-beanstalk.zip
```

---

## Step 2: Create the AWS Elastic Beanstalk App

In AWS Console:

1. Open `Elastic Beanstalk`
2. Click `Create application`
3. Application name: `stock-prediction-project`
4. Platform: `Python 3.11`
5. Upload code: choose `dist\aws-elastic-beanstalk.zip`

For environment type, use:

- `Single instance`

For instance size, use a free-tier eligible micro instance if your AWS account still has free-tier eligibility.

---

## Step 3: Set Environment Variables

In Elastic Beanstalk environment configuration, set:

- `BOOTSTRAP_ADMIN_USERNAME`
- `BOOTSTRAP_ADMIN_PASSWORD`

Already handled in project config:

- `APP_STORAGE_ROOT=/var/app/data`
- `FLASK_DEBUG=0`
- `BOOTSTRAP_ADMIN_SYNC=true`

---

## Step 4: Deploy and Test

After deployment:

1. Open the Elastic Beanstalk URL
2. Go to `/admin/login`
3. Log in with the bootstrap username/password you set
4. Test dashboard and admin actions

---

## Important Notes

- Free-tier eligibility on AWS depends on your AWS account status and signup date
- Elastic Beanstalk itself has no extra charge, but the underlying AWS resources do
- If AWS creates a load balancer or larger instance type, that can cost money
- For the lowest-cost setup, keep it as a single-instance environment

---

## If You Want Me to Help Further

After you build the bundle, I can walk you through the exact Elastic Beanstalk console screens step by step.
