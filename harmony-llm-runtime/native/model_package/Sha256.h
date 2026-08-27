#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace hllm {

/*
 * Minimal SHA-256 used to verify HLLM artifact integrity. The package is data,
 * not code; checksums must be validated before a model is installed. A real
 * device build may replace this with a platform crypto library (OpenSSL /
 * mbedTLS) without changing callers.
 */
std::string Sha256Hex(const std::uint8_t* data, std::size_t len);
std::string Sha256Hex(const std::string& data);

}  // namespace hllm
