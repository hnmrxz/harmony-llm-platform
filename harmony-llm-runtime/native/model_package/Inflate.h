#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace hllm {

/*
 * Minimal DEFLATE (RFC 1951) decompressor, bundled so the runtime can unpack
 * a `.hllm` (a ZIP archive with deflated entries) without depending on an
 * external zlib on the host or the device.
 *
 * Supports stored, fixed-Huffman and dynamic-Huffman blocks. Returns false on
 * any malformed stream.
 */
bool Inflate(const std::uint8_t* data, std::size_t len, std::string& out);

}  // namespace hllm
