#!/usr/bin/env bash
# Install portable Blender 4.5 LTS for the "Build a scene with Astra and Blender" lab.
#
# Downloads the official Linux x64 archive, verifies its SHA-256, and extracts it under
# $ASTRA_SCENE_DIR (default ~/astra-scene). Installs the few X11/GL runtime libraries
# Blender needs if any are missing, then confirms Blender runs and writes blender.env.
#
#   bash install_blender.sh
#   source ~/astra-scene/blender.env     # sets BLENDER_BIN
set -euo pipefail

BLENDER_VERSION="4.5.15"
BLENDER_SERIES="4.5"
BLENDER_SHA256="0d2b8bb2efe97e6c8eb57de6b7322f8800e1984e3125b60decca5f6771e4773e"
ASTRA_SCENE_DIR="${ASTRA_SCENE_DIR:-$HOME/astra-scene}"

ARCHIVE="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
URL="https://download.blender.org/release/Blender${BLENDER_SERIES}/${ARCHIVE}"
BLENDER_HOME="${ASTRA_SCENE_DIR}/blender-${BLENDER_VERSION}-linux-x64"
BLENDER_BIN="${BLENDER_HOME}/blender"

# Runtime libraries the official Linux build links against. Ubuntu desktop images
# usually have them already; only missing packages are installed.
LIBS=(libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libgl1 libegl1)

mkdir -p "$ASTRA_SCENE_DIR"
cd "$ASTRA_SCENE_DIR"

missing=()
for pkg in "${LIBS[@]}"; do
  dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Installing missing libraries: ${missing[*]}"
  sudo apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq "${missing[@]}"
fi

if [ ! -x "$BLENDER_BIN" ]; then
  echo "Downloading ${ARCHIVE} (about 380 MB)"
  curl -fL --retry 3 -o "$ARCHIVE" "$URL"
  echo "${BLENDER_SHA256}  ${ARCHIVE}" | sha256sum -c -
  tar -xJf "$ARCHIVE"
  rm -f "$ARCHIVE"
fi

version_line="$("$BLENDER_BIN" --background --factory-startup --version 2>/dev/null | head -n 1)"
case "$version_line" in
  "Blender ${BLENDER_VERSION}"*) ;;
  *) echo "FAIL - unexpected Blender version: ${version_line:-none}"; exit 1 ;;
esac

printf 'export ASTRA_SCENE_DIR=%q\nexport BLENDER_BIN=%q\n' \
  "$ASTRA_SCENE_DIR" "$BLENDER_BIN" > "${ASTRA_SCENE_DIR}/blender.env"

echo "PASS - ${version_line} at ${BLENDER_BIN}"
echo "PASS - wrote ${ASTRA_SCENE_DIR}/blender.env"
