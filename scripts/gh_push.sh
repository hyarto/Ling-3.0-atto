#!/usr/bin/env bash
# Usage: GITHUB_TOKEN=... scripts/gh_push.sh "commit message"
# The token is only passed through an http header for this one push; it is never written to .git/config.
set -euo pipefail
git add -A && git commit -q -m "$1" || echo "nothing to commit"
B64=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)
git -c http.extraheader="Authorization: Basic $B64" push -q https://github.com/hyarto/Ling-3.0-atto.git HEAD:main
echo "pushed: $(git log --oneline | head -1)"
