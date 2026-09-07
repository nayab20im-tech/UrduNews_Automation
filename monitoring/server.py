"""
monitoring/server.py
====================
Zero-dependency web backend for Section 7 — Monitoring & Dashboard.

Serves the dashboard over Python's stdlib ``http.server`` (the offline
fallback when Streamlit/FastAPI are not installed) with:

* ``GET /``              -- self-contained HTML dashboard (auto-refresh)
* ``GET /api/report``    -- the full aggregated report as JSON
* ``GET /api/status``    -- pipeline status section
* ``GET /api/news``      -- latest news preview
* ``GET /api/uploads``   -- upload/distribution status
* ``GET /api/analytics`` -- analytics overview
* ``GET /api/health``    -- storage health probes
* ``GET /api/logs``      -- recent system logs
* ``GET /api/syscheck``  -- Section 8 system-requirements check

The HTML page renders DOM via ``textContent`` only (no innerHTML for
data) so user-controlled news titles can never inject markup.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Tuple

from data_acquisition.utils.logger import get_logger

from monitoring import config as mcfg
from monitoring.dashboard_data import DashboardData

logger = get_logger("monitoring.server")


class DashboardHandler(BaseHTTPRequestHandler):
    """Handler with a shared :class:`DashboardData` injected by the factory."""

    data: DashboardData  # set by create_dashboard_server
    refresh_seconds: int = mcfg.DASHBOARD_REFRESH_SECONDS

    ROUTES = {
        "/api/report": lambda d: d.full_report(),
        "/api/status": lambda d: d.pipeline_status(),
        "/api/news": lambda d: {"latest_news": d.latest_news()},
        "/api/uploads": lambda d: d.upload_status(),
        "/api/analytics": lambda d: d.analytics_overview(),
        "/api/health": lambda d: d.storage_health(),
        "/api/logs": lambda d: {"recent_logs": d.recent_logs()},
        "/api/syscheck": lambda d: {"system_check": d.system_check()},
    }

    def do_GET(self) -> None:  # noqa: N802 - http.server convention
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_html(_DASHBOARD_HTML.replace(
                "__REFRESH_SECONDS__", str(self.refresh_seconds)))
            return
        route = self.ROUTES.get(path)
        if route is None:
            self._send_json({"error": f"unknown route: {path}"}, status=404)
            return
        try:
            self._send_json(route(self.data))
        except Exception as exc:  # noqa: BLE001 - never drop the request
            logger.exception("Dashboard route %s failed", path)
            self._send_json({"error": str(exc)}, status=500)

    # -- transport helpers -------------------------------------------------
    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2,
                          default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
        logger.debug("HTTP %s", fmt % args)


def create_dashboard_server(
    data: DashboardData,
    host: str = mcfg.DASHBOARD_HOST,
    port: int = mcfg.DASHBOARD_PORT,
    refresh_seconds: int = mcfg.DASHBOARD_REFRESH_SECONDS,
) -> Tuple[ThreadingHTTPServer, int]:
    """Build (but don't start) the server; returns ``(httpd, bound_port)``."""
    handler = type("BoundDashboardHandler", (DashboardHandler,),
                   {"data": data, "refresh_seconds": refresh_seconds})
    httpd = ThreadingHTTPServer((host, port), handler)
    logger.info("Dashboard server ready on http://%s:%s",
                host, httpd.server_address[1])
    return httpd, httpd.server_address[1]


def serve(
    data: DashboardData,
    host: str = mcfg.DASHBOARD_HOST,
    port: int = mcfg.DASHBOARD_PORT,
    refresh_seconds: int = mcfg.DASHBOARD_REFRESH_SECONDS,
) -> None:
    """Blocking serve loop (Ctrl-C stops cleanly)."""
    httpd, bound_port = create_dashboard_server(data, host, port,
                                                refresh_seconds)
    print(f"UrduNewsAI dashboard: http://{host}:{bound_port}  "
          f"(Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# Dashboard page (single self-contained file; no external assets)
# ---------------------------------------------------------------------------
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UrduNewsAI — Operations Dashboard</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --line:#334155; --text:#e2e8f0;
          --dim:#94a3b8; --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:var(--bg);
         color:var(--text); }
  header { padding:16px 24px; border-bottom:1px solid var(--line);
           display:flex; justify-content:space-between; align-items:center; }
  header h1 { font-size:18px; margin:0; }
  header .meta { color:var(--dim); font-size:12px; text-align:right; }
  main { padding:20px 24px; display:grid; gap:16px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
          gap:12px; }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:8px; padding:14px; }
  .card h3 { margin:0 0 8px; font-size:13px; color:var(--dim);
             text-transform:uppercase; letter-spacing:.05em; }
  .stat { font-size:26px; font-weight:700; }
  .sub { color:var(--dim); font-size:12px; margin-top:4px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
           vertical-align:top; }
  th { color:var(--dim); font-weight:600; font-size:12px; }
  .pill { display:inline-block; padding:1px 8px; border-radius:10px;
          font-size:11px; background:var(--line); }
  .pill.ok { background:var(--ok); color:#052e16; }
  .pill.warn { background:var(--warn); color:#451a03; }
  .pill.bad, .pill.missing, .pill.failed { background:var(--bad); color:#450a0a; }
  .urdu { direction:rtl; font-size:14px; }
  .muted { color:var(--dim); }
</style>
</head>
<body>
<header>
  <h1>UrduNewsAI — Operations Dashboard</h1>
  <div class="meta"><div id="generated">…</div>
  <div>auto-refresh every __REFRESH_SECONDS__s</div></div>
</header>
<main>
  <section class="grid" id="counters"></section>
  <section class="card"><h3>Pipeline Status — recent orchestration runs</h3>
    <table id="runs"></table></section>
  <section class="card"><h3>Latest News Preview</h3>
    <table id="news"></table></section>
  <section class="card"><h3>Upload Status</h3>
    <table id="uploads"></table></section>
  <section class="card"><h3>Analytics Overview</h3>
    <div id="analytics"></div></section>
  <section class="grid" id="health"></section>
  <section class="card"><h3>System Requirements Check</h3>
    <table id="syscheck"></table></section>
  <section class="card"><h3>Recent Logs</h3><table id="logs"></table></section>
</main>
<script>
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
function table(node, header, rows, rowClass) {
  clear(node);
  const thead = el('thead'), trh = document.createElement('tr');
  header.forEach(h => trh.appendChild(el('th', null, h)));
  thead.appendChild(trh); node.appendChild(thead);
  const tbody = el('tbody');
  rows.forEach(r => {
    const tr = document.createElement('tr');
    if (rowClass) tr.className = rowClass(r) || '';
    r.cells.forEach(c => tr.appendChild(
      c.node || el('td', c.cls, c.text)));
    tbody.appendChild(tr);
  });
  node.appendChild(tbody);
}
function counter(node, label, value, sub) {
  const card = el('div', 'card');
  card.appendChild(el('h3', null, label));
  card.appendChild(el('div', 'stat', value));
  if (sub) card.appendChild(el('div', 'sub', sub));
  node.appendChild(card);
}
function render(d) {
  document.getElementById('generated').textContent =
    'report generated: ' + (d.generated_at || '');
  const ps = d.pipeline_status || {};
  const raw = ps.raw || {}, proc = ps.processed || {},
        dist = ps.distribution || {}, media = ps.media || {};
  const counters = document.getElementById('counters'); clear(counters);
  counter(counters, 'Raw articles', raw.total_articles ?? '—',
    Object.entries(raw.by_status || {}).map(([k,v]) => k + ': ' + v).join(' · '));
  counter(counters, 'Processed', proc.articles ?? '—',
    (proc.translated ?? 0) + ' translated · ' + (proc.classified ?? 0) + ' classified');
  counter(counters, 'Audio (TTS)', proc.with_tts_audio ?? '—',
    (proc.with_processed_audio ?? 0) + ' post-processed');
  counter(counters, 'Jobs', dist.jobs ?? '—',
    (dist.published ?? 0) + ' published on YouTube');
  const mediaKinds = Object.entries(media.by_kind || {});
  const mediaTotal = media.total_assets ??
    mediaKinds.reduce((a, [k, v]) => a + v.files, 0);
  counter(counters, 'Media assets', mediaTotal,
    mediaKinds.map(([k, v]) => k + ': ' + v.files).join(' · '));

  table(document.getElementById('runs'),
    ['Run', 'Stage', 'Status', 'Attempts', 'Started', 'Error'],
    (ps.recent_runs || []).map(r => ({cells: [
      {text: r.run_id || r.error || ''}, {text: r.stage || ''},
      {text: r.status || ''}, {text: r.attempts ?? ''},
      {text: r.started_at || ''}, {text: r.error || ''}]})),
    r => 'pill ' + (r.status || ''));

  table(document.getElementById('news'),
    ['ID', 'Title', 'Urdu Title', 'Category', 'Bias', 'Verdict', 'Audio', 'YouTube'],
    (d.latest_news || []).map(n => ({cells: [
      {text: n.article_id ?? n.error ?? ''}, {text: n.title || ''},
      {node: el('td', 'urdu', n.urdu_title || '')},
      {text: n.category || ''}, {text: n.bias_label || ''},
      {text: n.verification_verdict || ''},
      {text: n.has_tts_audio ? 'yes' : 'no'},
      {text: n.youtube_status || ''}]})));

  const up = d.upload_status || {};
  table(document.getElementById('uploads'),
    ['ID', 'Title', 'Status', 'YouTube', 'Retries', 'Updated'],
    (up.recent || []).map(j => ({cells: [
      {text: j.article_id ?? ''}, {text: j.headline || ''},
      {text: j.status || ''}, {text: j.youtube_status || ''},
      {text: j.retry_count ?? 0}, {text: j.updated_at || ''}]})));

  const an = d.analytics_overview || {};
  const anNode = document.getElementById('analytics'); clear(anNode);
  const tot = an.totals || {};
  const row1 = el('div', 'sub',
    'videos: ' + (tot.videos ?? 0) + ' · views: ' + (tot.views ?? 0) +
    ' · likes: ' + (tot.likes ?? 0) + ' · comments: ' + (tot.comments ?? 0) +
    ' · shares: ' + (tot.shares ?? 0));
  anNode.appendChild(row1);
  const tbl = el('table');
  table(tbl, ['Platform', 'Videos', 'Views', 'Likes', 'Comments', 'Shares'],
    Object.entries(an.by_platform || {}).map(([p, a]) => ({cells: [
      {text: p}, {text: a.videos}, {text: a.views}, {text: a.likes},
      {text: a.comments}, {text: a.shares}]})));
  anNode.appendChild(tbl);

  const healthNode = document.getElementById('health'); clear(healthNode);
  Object.entries(d.storage_health || {}).forEach(([name, probe]) => {
    const card = el('div', 'card');
    card.appendChild(el('h3', null, name));
    const ok = probe && probe.ok;
    const div = el('div', 'stat', ok ? 'OK' : 'ERR');
    div.style.color = ok ? 'var(--ok)' : 'var(--bad)';
    card.appendChild(div);
    card.appendChild(el('div', 'sub', probe && (probe.error || probe.path || '')));
    healthNode.appendChild(card);
  });

  table(document.getElementById('syscheck'),
    ['Requirement', 'Status', 'Detected', 'Recommended'],
    (d.system_check || []).map(c => ({cells: [
      {text: c.name}, {node: el('td', null, '')},
      {text: c.detail}, {text: c.recommended}]}), null));
  // status pills
  document.querySelectorAll('#syscheck tbody tr').forEach((tr, i) => {
    const c = (d.system_check || [])[i];
    const td = tr.children[1];
    td.appendChild(el('span', 'pill ' + (c ? c.status : ''), c ? c.status : ''));
  });

  table(document.getElementById('logs'),
    ['Time', 'Level', 'Stage', 'Message'],
    (d.recent_logs || []).map(l => ({cells: [
      {text: l.logged_at || ''}, {node: el('td', null, '')},
      {text: l.stage || l.logger || ''}, {text: l.message || l.error || ''}]})));
  document.querySelectorAll('#logs tbody tr').forEach((tr, i) => {
    const l = (d.recent_logs || [])[i];
    const lvl = (l && l.level || '').toLowerCase();
    const cls = lvl === 'error' ? 'bad' : (lvl === 'warning' ? 'warn' : '');
    tr.children[1].appendChild(el('span', 'pill ' + cls, l ? l.level : ''));
  });
}
function refresh() {
  fetch('/api/report').then(r => r.json()).then(render)
    .catch(e => console.error('refresh failed', e));
}
refresh();
setInterval(refresh, __REFRESH_SECONDS__ * 1000);
</script>
</body>
</html>
"""
