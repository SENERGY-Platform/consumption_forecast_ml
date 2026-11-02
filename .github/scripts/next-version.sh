#!/usr/bin/env bash
# Prints the version of HEAD: the version tag HEAD already carries, else the last version
# tag reachable from HEAD bumped by the conventional commits since, else v2.0.0.
# The image continues the one of SENERGY-Platform/consumption_forecast, which got up to
# v1.0.1, so starting at v1.0.0 would overwrite its immutable tags.
set -euo pipefail

pattern='v[0-9]*.[0-9]*.[0-9]*'

current=$(git tag --points-at HEAD --list "$pattern" | sort -V | tail -n 1)
if [ -n "$current" ]; then
  echo "$current"
  exit 0
fi

last=$(git describe --tags --abbrev=0 --match "$pattern" 2>/dev/null || true)
if [ -z "$last" ]; then
  echo "v2.0.0"
  exit 0
fi

IFS=. read -r major minor patch <<< "${last#v}"
log=$(git log --format='%s%n%b' "$last..HEAD")
if grep -qE '^[a-z]+(\([^)]*\))?!:|^BREAKING[ -]CHANGE:' <<< "$log"; then
  echo "v$((major + 1)).0.0"
elif grep -qE '^feat(\([^)]*\))?:' <<< "$log"; then
  echo "v$major.$((minor + 1)).0"
else
  echo "v$major.$minor.$((patch + 1))"
fi
