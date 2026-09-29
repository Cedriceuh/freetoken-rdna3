#!/usr/bin/env bash
# Pre-publication scan: personal data in the tracked files and in the commit messages / authors being published.
#   rdna3/tools/privacy_scan.sh [extra-pattern ...]        BASE=<ref> (default upstream/main) bounds the history scan
# Built-in patterns: home paths, e-mail addresses other than GitHub noreply ones, common secret shapes (API keys,
# Discord / Brave tokens), AI attribution lines. Your own names (machine, employer, private projects) go in as extra
# patterns, or one extended regex per line in the file named by PRIVACY_PATTERNS_FILE (keep that file outside the
# repository). Prints matches; exit 1 if any.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
base="${BASE:-upstream/main}"
patterns=(
  '/home/[a-z][a-z0-9_-]*/'
  '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
  '(sk|pk|api|key|token)[_-]?[A-Za-z0-9]{24,}'
  '[MN][A-Za-z0-9]{23}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}'
  'BSA[A-Za-z0-9_-]{20,}'
  "$@"
)
if [ -n "${PRIVACY_PATTERNS_FILE:-}" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) ;; *) patterns+=("$line") ;; esac
  done < "$PRIVACY_PATTERNS_FILE"
fi
# AI attribution: checked before the allow-list, which would otherwise let "noreply@..." addresses through
trailers='(Co-Authored-By|Co-authored-by|Assisted-by|Generated-by): |Generated with \['
# addresses that are fine to publish: GitHub noreply ones and placeholders
allow='users\.noreply\.github\.com|@example\.(com|org)|build@localhost'
found=0
for p in "${patterns[@]}"; do
  hits=$(git grep -nIE "$p" -- . ':!*.csv' 2>/dev/null | grep -vE "$allow" | grep -v 'rdna3/tools/privacy_scan.sh')
  if [ -n "$hits" ]; then echo "== files: $p"; echo "$hits" | head -20; found=1; fi
done
hits=$(git grep -nIE "$trailers" -- . 2>/dev/null | grep -v 'rdna3/tools/privacy_scan.sh')
if [ -n "$hits" ]; then echo "== files: AI attribution"; echo "$hits" | head -20; found=1; fi
if git rev-parse -q --verify "$base" >/dev/null; then
  echo "== history since $base: authors / committers"
  git log --format='%an <%ae> | %cn <%ce>' "$base"..HEAD | sort | uniq -c
  msgs=$(git log --format=%B "$base"..HEAD | grep -nIE "$(IFS='|'; echo "${patterns[*]}")" | grep -vE "$allow")
  if [ -n "$msgs" ]; then echo "== commit messages"; echo "$msgs" | head -20; found=1; fi
  msgs=$(git log --format=%B "$base"..HEAD | grep -nIE "$trailers")
  if [ -n "$msgs" ]; then echo "== commit messages: AI attribution"; echo "$msgs" | head -20; found=1; fi
else
  echo "(no $base ref: history not scanned; set BASE=<ref>)"
fi
exit $found
