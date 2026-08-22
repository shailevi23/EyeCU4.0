"""
Tiny local labeling server for the team-assignment development benchmark.
stdlib only (http.server) -- no new dependency.

Serves this directory statically (index.html, selection_manifest.json, and
the crops/ tree written by ../select_tracks.py), plus two endpoints:

  GET  /labels   -> current labels.json contents (or {} if not yet created)
  POST /save     -> body {"match_id": ..., "track_id": ..., "label": ...}
                    autosaves into labels.json immediately (one label per
                    match_id:track_id key; re-labeling overwrites)

Run:  python server.py [port]   (default port 8765)
Then open http://localhost:8765/index.html
"""
import http.server
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(HERE, 'labels.json')

VALID_LABELS = {'TEAM_A', 'TEAM_B', 'MIXED_TRACK', 'AMBIGUOUS'}


def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_labels(labels):
    tmp = LABELS_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    os.replace(tmp, LABELS_PATH)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def do_GET(self):
        if self.path == '/labels':
            body = json.dumps(load_labels()).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        if self.path == '/save':
            length = int(self.headers.get('Content-Length', 0))
            try:
                payload = json.loads(self.rfile.read(length))
                match_id = str(payload['match_id'])
                track_id = str(payload['track_id'])
                label = str(payload['label'])
                if label not in VALID_LABELS:
                    raise ValueError(f"invalid label {label!r}")
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
                return
            labels = load_labels()
            labels[f"{match_id}:{track_id}"] = label
            save_labels(labels)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass  # keep the console quiet; labels.json is the record


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = http.server.HTTPServer(('127.0.0.1', port), Handler)
    print(f"Label UI running at http://127.0.0.1:{port}/index.html")
    print(f"Autosaving to {LABELS_PATH}")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
