#include "Inflate.h"

#include <array>

namespace hllm {

namespace {

class BitReader {
public:
    BitReader(const std::uint8_t* data, std::size_t len) : data_(data), len_(len) {}

    bool ReadBit(unsigned& bit) {
        if (byteIndex_ >= len_) {
            return false;
        }
        if (bitIndex_ == 0) {
            current_ = data_[byteIndex_];
        }
        bit = (current_ >> bitIndex_) & 1u;
        ++bitIndex_;
        if (bitIndex_ == 8) {
            bitIndex_ = 0;
            ++byteIndex_;
        }
        return true;
    }

    bool ReadBits(unsigned count, unsigned& value) {
        value = 0;
        for (unsigned i = 0; i < count; ++i) {
            unsigned bit;
            if (!ReadBit(bit)) {
                return false;
            }
            value |= bit << i;
        }
        return true;
    }

    bool Align() {
        bitIndex_ = 0;
        return true;
    }

private:
    const std::uint8_t* data_;
    std::size_t len_;
    std::size_t byteIndex_ = 0;
    unsigned bitIndex_ = 0;
    unsigned current_ = 0;
};

// Length/distance lookup code tables from RFC 1951.
constexpr std::uint16_t kLengthBase[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27,
    31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
};
constexpr unsigned kLengthExtra[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
    2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
};
constexpr std::uint16_t kDistBase[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129,
    193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
};
constexpr unsigned kDistExtra[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
    6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
};

struct Huffman {
    // Canonical Huffman decode table: (code, length) keyed by symbol index.
    std::array<std::uint16_t, 288> codes{};
    std::array<std::uint8_t, 288> lengths{};
    int maxLen = 0;
    int numSymbols = 0;
};

bool BuildHuffman(const std::uint8_t* lengths, unsigned n, Huffman& huffman) {
    unsigned maxLen = 0;
    std::array<unsigned, 16> blCount{};
    for (unsigned i = 0; i < n; ++i) {
        unsigned len = lengths[i];
        if (len > 15) {
            return false;
        }
        if (len > 0) {
            ++blCount[len];
        }
        if (len > maxLen) {
            maxLen = len;
        }
    }
    huffman.maxLen = static_cast<int>(maxLen);
    huffman.numSymbols = static_cast<int>(n);

    // Compute canonical codes.
    std::array<unsigned, 16> nextCode{};
    unsigned code = 0;
    for (unsigned l = 1; l <= 15; ++l) {
        code = (code + blCount[l - 1]) << 1;
        nextCode[l] = code;
    }
    for (unsigned i = 0; i < n; ++i) {
        unsigned len = lengths[i];
        if (len == 0) {
            huffman.codes[i] = 0;
            huffman.lengths[i] = 0;
            continue;
        }
        huffman.codes[i] = static_cast<std::uint16_t>(nextCode[len]);
        ++nextCode[len];
        huffman.lengths[i] = static_cast<std::uint8_t>(len);
    }
    return true;
}

bool DecodeSymbol(BitReader& reader, const Huffman& huffman, unsigned& symbol) {
    unsigned code = 0;
    for (int len = 1; len <= huffman.maxLen; ++len) {
        unsigned bit;
        if (!reader.ReadBit(bit)) {
            return false;
        }
        code = (code << 1) | bit;
        for (int i = 0; i < huffman.numSymbols; ++i) {
            if (huffman.lengths[i] == len && huffman.codes[i] == code) {
                symbol = static_cast<unsigned>(i);
                return true;
            }
        }
    }
    return false;
}

bool InflateBlock(BitReader& reader, Huffman& litLen, Huffman& dist, std::string& out) {
    // A dynamic-Huffman block re-builds the tables; that logic lives in the
    // caller via ReadDynamicTables(). Keep this function for the in-loop symbol
    // decode which does not depend on block type.
    while (true) {
        unsigned sym;
        if (!DecodeSymbol(reader, litLen, sym)) {
            return false;
        }
        if (sym < 256) {
            out.push_back(static_cast<char>(sym));
            continue;
        }
        if (sym == 256) {
            return true;  // end-of-block
        }
        unsigned lengthIndex = sym - 257;
        if (lengthIndex >= 29) {
            return false;
        }
        unsigned extraLength;
        if (!reader.ReadBits(kLengthExtra[lengthIndex], extraLength)) {
            return false;
        }
        unsigned length = kLengthBase[lengthIndex] + extraLength;

        unsigned distSymbol;
        if (!DecodeSymbol(reader, dist, distSymbol)) {
            return false;
        }
        if (distSymbol >= 30) {
            return false;
        }
        unsigned extraDist;
        if (!reader.ReadBits(kDistExtra[distSymbol], extraDist)) {
            return false;
        }
        unsigned distance = kDistBase[distSymbol] + extraDist;
        if (distance == 0 || distance > out.size()) {
            return false;
        }
        for (unsigned i = 0; i < length; ++i) {
            char c = out[out.size() - distance];
            out.push_back(c);
        }
    }
}

bool ReadDynamicTables(BitReader& reader, Huffman& litLen, Huffman& dist) {
    unsigned hlit, hdist, hclen;
    if (!reader.ReadBits(5, hlit) || !reader.ReadBits(5, hdist) || !reader.ReadBits(4, hclen)) {
        return false;
    }
    hlit += 257;
    hdist += 1;
    hclen += 4;
    if (hlit > 286 || hdist > 30 || hclen > 19) {
        return false;
    }

    static const unsigned order[19] = {
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    };
    std::array<std::uint8_t, 19> clLengths{};
    for (unsigned i = 0; i < hclen; ++i) {
        unsigned v;
        if (!reader.ReadBits(3, v)) {
            return false;
        }
        clLengths[order[i]] = static_cast<std::uint8_t>(v);
    }
    Huffman cl;
    if (!BuildHuffman(clLengths.data(), 19, cl)) {
        return false;
    }

    std::array<std::uint8_t, 288 + 30> all{};
    unsigned idx = 0;
    while (idx < hlit + hdist) {
        unsigned sym;
        if (!DecodeSymbol(reader, cl, sym)) {
            return false;
        }
        if (sym < 16) {
            all[idx++] = static_cast<std::uint8_t>(sym);
        } else if (sym == 16) {
            if (idx == 0) {
                return false;
            }
            unsigned rep;
            if (!reader.ReadBits(2, rep)) {
                return false;
            }
            rep += 3;
            std::uint8_t prev = all[idx - 1];
            for (unsigned i = 0; i < rep; ++i) {
                all[idx++] = prev;
            }
        } else if (sym == 17) {
            unsigned rep;
            if (!reader.ReadBits(3, rep)) {
                return false;
            }
            rep += 3;
            for (unsigned i = 0; i < rep; ++i) {
                all[idx++] = 0;
            }
        } else if (sym == 18) {
            unsigned rep;
            if (!reader.ReadBits(7, rep)) {
                return false;
            }
            rep += 11;
            for (unsigned i = 0; i < rep; ++i) {
                all[idx++] = 0;
            }
        } else {
            return false;
        }
        if (idx > 286 + 30) {
            return false;
        }
    }

    if (!BuildHuffman(all.data(), hlit, litLen)) {
        return false;
    }
    if (!BuildHuffman(all.data() + hlit, hdist, dist)) {
        return false;
    }
    return true;
}

}  // namespace

bool Inflate(const std::uint8_t* data, std::size_t len, std::string& out) {
    BitReader reader(data, len);
    out.clear();
    while (true) {
        unsigned bit;
        if (!reader.ReadBit(bit)) {
            return false;
        }
        bool final = (bit == 1);
        unsigned type;
        // Two bits for block type.
        unsigned b1, b2;
        if (!reader.ReadBit(b1) || !reader.ReadBit(b2)) {
            return false;
        }
        type = (b1) | (b2 << 1);
        if (type == 0) {
            // Stored.
            reader.Align();
            unsigned lenVal, nlen;
            if (!reader.ReadBits(16, lenVal) || !reader.ReadBits(16, nlen)) {
                return false;
            }
            if ((lenVal ^ 0xFFFF) != nlen) {
                return false;
            }
            for (unsigned i = 0; i < lenVal; ++i) {
                // Read a full byte.
                unsigned v;
                if (!reader.ReadBits(8, v)) {
                    return false;
                }
                out.push_back(static_cast<char>(v));
            }
        } else if (type == 1 || type == 2) {
            Huffman litLen, dist;
            if (type == 2) {
                if (!ReadDynamicTables(reader, litLen, dist)) {
                    return false;
                }
            } else {
                // Fixed Huffman: literal/length 0-143 (8 bits), 144-255 (9),
                // 256-279 (7), 280-287 (8). Distance 0-31 (5 bits).
                std::array<std::uint8_t, 288> litLengths{};
                for (unsigned i = 0; i <= 143; ++i) {
                    litLengths[i] = 8;
                }
                for (unsigned i = 144; i <= 255; ++i) {
                    litLengths[i] = 9;
                }
                for (unsigned i = 256; i <= 279; ++i) {
                    litLengths[i] = 7;
                }
                for (unsigned i = 280; i <= 287; ++i) {
                    litLengths[i] = 8;
                }
                std::array<std::uint8_t, 30> distLengths{};
                distLengths.fill(5);
                if (!BuildHuffman(litLengths.data(), 288, litLen) ||
                    !BuildHuffman(distLengths.data(), 30, dist)) {
                    return false;
                }
            }
            if (!InflateBlock(reader, litLen, dist, out)) {
                return false;
            }
        } else {
            return false;  // reserved block type
        }
        if (final) {
            return true;
        }
    }
}

}  // namespace hllm
