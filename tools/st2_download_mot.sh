#!/usr/bin/env bash
# Fetch the SoccerTrack v2 MOT component, once a Hugging Face credential exists.
#
# The first download attempt left mot-...zip at 158 bytes -- an empty archive --
# so the MOT half of the audit could not be answered at all. This re-runs it
# properly.
#
# Two things blocked it here, and only one of them is fixed in this file:
#
#   SSL   This machine sits behind a TLS-intercepting proxy whose root CA is in
#         the Windows certificate store but not in certifi's bundle. Python's
#         stdlib urllib trusts it, `requests` (and therefore huggingface_hub)
#         does not. ca_bundle_system.pem is certifi plus the Windows store, and
#         pointing REQUESTS_CA_BUNDLE at it makes the CLI work. Nothing is
#         installed and no EyeCU dependency is touched.
#
#   AUTH  atomscott/soccertrack-v2 is GATED. It needs a token AND an approved
#         access request. Neither can be done from here: `hf auth login` is
#         interactive, and a token must never be pasted into an agent
#         transcript. Do this yourself, once:
#
#             hf auth login                                  (interactive)
#         or  export HF_TOKEN=...                            (your shell only)
#
#         and request access at
#             https://huggingface.co/datasets/atomscott/soccertrack-v2
#
# Then run this script. It refuses to start rather than emitting a half-download.

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
EXT="$REPO_ROOT/EyeCU_external_data"
CLONE="$EXT/SoccerTrack-v2"
DEST="${1:-$EXT/soccertrack_mot}"

BUNDLE="$EXT/ca_bundle_system.pem"
if [[ ! -f "$BUNDLE" ]]; then
  echo "regenerating the CA bundle"
  python - <<'PY'
import ssl, certifi, pathlib
out = pathlib.Path('EyeCU_external_data/ca_bundle_system.pem')
pem = [pathlib.Path(certifi.where()).read_text(encoding='utf-8')]
for store in ('ROOT', 'CA'):
    try:
        for cert, enc, _ in ssl.enum_certificates(store):
            if enc == 'x509_asn':
                pem.append(ssl.DER_cert_to_PEM_cert(cert))
    except Exception:
        pass
out.write_text('\n'.join(pem), encoding='utf-8')
PY
fi
export REQUESTS_CA_BUNDLE="$BUNDLE" CURL_CA_BUNDLE="$BUNDLE" SSL_CERT_FILE="$BUNDLE"
export PATH="$REPO_ROOT/eye_env/Scripts:$PATH"

# `hf auth whoami` prints "Not logged in" and still exits 0, so an exit-code
# check passes when logged out -- which is why the first run of this script sailed
# past its own guard and into a 401. Upstream scripts/download.sh has the same
# flaw. Check what it SAYS, not what it returns.
WHO="$(hf auth whoami 2>/dev/null || true)"
if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "$WHO" || "$WHO" == *"Not logged in"* ]]; then
  echo "NOT AUTHENTICATED -- nothing downloaded." >&2
  echo "  run 'hf auth login' (or export HF_TOKEN) and make sure your access" >&2
  echo "  request at https://huggingface.co/datasets/atomscott/soccertrack-v2" >&2
  echo "  has been approved, then re-run this script." >&2
  exit 1
fi

if [[ ! -d "$CLONE" ]]; then
  echo "SoccerTrack-v2 checkout missing at $CLONE" >&2
  exit 1
fi

echo "downloading mot/* -> $DEST"
bash "$CLONE/scripts/download.sh" --dest "$DEST" --include "mot/*"

echo
echo "downloaded:"
find "$DEST" -type f -printf '%12s  %p\n' | sort -k2 | head -50
