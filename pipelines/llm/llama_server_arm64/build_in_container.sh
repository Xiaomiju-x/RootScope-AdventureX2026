#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=/src
OUTPUT_ROOT=/out
TOOLS_ROOT=/tools
BUILD_ROOT="$OUTPUT_ROOT/build"
STAGE_ROOT="$OUTPUT_ROOT/stage"

test -r "$SOURCE_ROOT/CMakeLists.txt"
test -r "$SOURCE_ROOT/LICENSE"
test -r "$TOOLS_ROOT/aarch64-linux-gnu.cmake"

rm -rf "$BUILD_ROOT" "$STAGE_ROOT"
mkdir -p "$BUILD_ROOT" "$STAGE_ROOT/bin" "$STAGE_ROOT/licenses" "$STAGE_ROOT/metadata"

export LC_ALL=C.UTF-8
export TZ=UTC
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:?SOURCE_DATE_EPOCH is required}"

cmake -S "$SOURCE_ROOT" -B "$BUILD_ROOT" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLS_ROOT/aarch64-linux-gnu.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -ffile-prefix-map=/src=/usr/src/llama.cpp-b9637 -fdebug-prefix-map=/src=/usr/src/llama.cpp-b9637" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -ffile-prefix-map=/src=/usr/src/llama.cpp-b9637 -fdebug-prefix-map=/src=/usr/src/llama.cpp-b9637" \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc -Wl,--build-id=sha1" \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_STATIC=ON \
  -DGGML_OPENMP=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH=armv8-a \
  -DGGML_BACKEND_DL=OFF \
  -DLLAMA_BUILD_NUMBER=9637 \
  -DLLAMA_BUILD_COMMIT=aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3 \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_OPENSSL=OFF \
  2>&1 | tee "$STAGE_ROOT/metadata/cmake_config.log"

cmake --build "$BUILD_ROOT" --target llama-server --parallel "$(nproc)" \
  2>&1 | tee "$STAGE_ROOT/metadata/build.log"

install -m 0755 "$BUILD_ROOT/bin/llama-server" "$STAGE_ROOT/bin/llama-server"
aarch64-linux-gnu-strip --strip-unneeded "$STAGE_ROOT/bin/llama-server"
install -m 0644 "$SOURCE_ROOT/LICENSE" "$STAGE_ROOT/licenses/llama.cpp-LICENSE"

{
  printf 'SOURCE_DATE_EPOCH=%s\n' "$SOURCE_DATE_EPOCH"
  printf 'BUILD_ARCH=%s\n' "$(uname -m)"
  printf 'BASE_OS='; . /etc/os-release; printf '%s %s\n' "$ID" "$VERSION_ID"
  printf 'CMAKE='; cmake --version | head -n 1
  printf 'C_COMPILER='; aarch64-linux-gnu-gcc --version | head -n 1
  printf 'CXX_COMPILER='; aarch64-linux-gnu-g++ --version | head -n 1
  printf 'BINUTILS='; aarch64-linux-gnu-ld --version | head -n 1
  printf 'NINJA='; ninja --version
  dpkg-query -W -f='${binary:Package}=${Version}\n' \
    binutils-aarch64-linux-gnu cmake g++-aarch64-linux-gnu gcc-aarch64-linux-gnu ninja-build
} > "$STAGE_ROOT/metadata/toolchain_versions.txt"

file "$STAGE_ROOT/bin/llama-server" > "$STAGE_ROOT/metadata/file.txt"
aarch64-linux-gnu-readelf -h "$STAGE_ROOT/bin/llama-server" > "$STAGE_ROOT/metadata/readelf_header.txt"
aarch64-linux-gnu-readelf -d "$STAGE_ROOT/bin/llama-server" > "$STAGE_ROOT/metadata/readelf_dynamic.txt"
aarch64-linux-gnu-readelf -n "$STAGE_ROOT/bin/llama-server" > "$STAGE_ROOT/metadata/readelf_notes.txt"
aarch64-linux-gnu-readelf --version-info "$STAGE_ROOT/bin/llama-server" > "$STAGE_ROOT/metadata/readelf_version_info.txt"
aarch64-linux-gnu-objdump -p "$STAGE_ROOT/bin/llama-server" > "$STAGE_ROOT/metadata/objdump_private_headers.txt"
