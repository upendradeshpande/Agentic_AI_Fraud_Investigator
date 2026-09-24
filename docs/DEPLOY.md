# Hosting the Fraud Investigator Tool

The app is a long-running Streamlit server (it keeps a WebSocket open to each browser), so it needs a host
that runs a persistent process. Recommended: **Streamlit Community Cloud** for a free reviewer link, or
**Google Cloud Run** for a production-like container deployment.

Before hosting, run `python3 verify_install.py` on GitHub's *Code → Download ZIP* to confirm the repo is complete.

## Option 1: Streamlit Community Cloud (free, about 10 minutes)

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. **Create app** → choose the repository, branch `main`, main file path `app/streamlit_app.py`.
3. **Advanced settings** → Python **3.12**. Optionally paste secrets (TOML format):

   ```toml
   # Everything is optional. Without a key the app runs fully in deterministic mode.
   APP_PASSWORD = "choose-a-demo-password"
   SESSION_QUESTION_LIMIT = "20"
   DEMO_NOTICE = "Demo: changes reset when the app restarts."
   # Only if you accept the cost risk (set a spend limit with your provider first):
   # LLM_PROVIDER = "anthropic"
   # ANTHROPIC_API_KEY = "sk-ant-..."
   ```

   Root-level secrets are loaded into the environment by `cockpit/config.py`, so they behave exactly like `.env`.
4. **Deploy**. You get a URL such as `https://your-app.streamlit.app`.

Free apps sleep after a period without visits and take about 30 seconds to wake. Open the link shortly before
reviewers are likely to use it.

## Option 2: Google Cloud Run (Docker, production-like)

Requires the gcloud CLI and a Google Cloud project with billing enabled (a demo stays within the free tier).

```bash
gcloud run deploy fraud-investigator \
  --source . --region us-central1 --allow-unauthenticated \
  --memory 1Gi --max-instances 1 --session-affinity --timeout 3600 \
  --set-env-vars "DEMO_NOTICE=Demo: changes reset when the app restarts.,SESSION_QUESTION_LIMIT=20"
# Optional, with the key stored in Secret Manager:
#   --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest --set-env-vars LLM_PROVIDER=anthropic
```

- `--max-instances 1`: the prototype keeps state in one SQLite file and in memory, so a single instance keeps
  every visitor on the same data. Production would use PostgreSQL and scale out.
- `--session-affinity` and `--timeout 3600` keep each browser's WebSocket on the same instance.
- The Dockerfile reads the port from `$PORT`, which Cloud Run sets automatically.

## Option 3: Render or Railway (Docker, from GitHub)

Create a new **Web Service** from the repository and choose **Docker**. Both services set `$PORT`, which the
Dockerfile uses. Add the same optional environment variables as above in the service settings.

## Why not Vercel?

Vercel runs short-lived serverless functions and static front ends. A Streamlit app needs a persistent server
with WebSocket connections, which Vercel does not provide. Vercel fits a production split instead: a React or
Next.js front end on Vercel calling a FastAPI backend (the existing `cockpit` service layer) hosted on Cloud Run.

## Public-link checklist

- **API keys.** Prefer no key in the host's secrets: the app works fully without one, and visitors can connect
  their own key in *Model connection*. That connection is private to their browser session (stored server-side
  in their session only, never logged, never shared with other visitors). If you add your own key, set a spend
  limit with the provider and use `SESSION_QUESTION_LIMIT`.
- **Access.** Set `APP_PASSWORD` to keep the link private to the people you send the password to.
- **Shared workspace.** All visitors share one demo queue: a decision made by one reviewer is visible to the
  next. *Data & rules → Reset* restores the original state.
- **Temporary storage.** Community Cloud and Cloud Run reset the local disk on restart. The 50 cases reload and
  re-assess automatically; investigator actions from before the restart are lost. Production would use PostgreSQL.
