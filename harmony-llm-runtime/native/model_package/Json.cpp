#include "Json.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace hllm {
namespace json {

namespace {

class Parser {
public:
    explicit Parser(const std::string& src) : src_(src) {}

    std::unique_ptr<Value> ParseDocument() {
        SkipWhitespace();
        auto value = ParseValue();
        if (value == nullptr) {
            return nullptr;
        }
        SkipWhitespace();
        if (pos_ != src_.size()) {
            return nullptr;  // trailing junk
        }
        return value;
    }

private:
    std::unique_ptr<Value> ParseValue() {
        if (pos_ >= src_.size()) {
            return nullptr;
        }
        char c = src_[pos_];
        if (c == '{') {
            return ParseObject();
        }
        if (c == '[') {
            return ParseArray();
        }
        if (c == '"') {
            return ParseStringValue();
        }
        if (c == 't') {
            return ParseLiteral("true", Value::Bool(true));
        }
        if (c == 'f') {
            return ParseLiteral("false", Value::Bool(false));
        }
        if (c == 'n') {
            return ParseLiteral("null", Value::Null());
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return ParseNumber();
        }
        return nullptr;
    }

    std::unique_ptr<Value> ParseLiteral(const char* literal, Value v) {
        std::size_t n = std::strlen(literal);
        if (src_.compare(pos_, n, literal) != 0) {
            return nullptr;
        }
        pos_ += n;
        return std::make_unique<Value>(std::move(v));
    }

    std::unique_ptr<Value> ParseNumber() {
        std::size_t start = pos_;
        if (pos_ < src_.size() && src_[pos_] == '-') {
            ++pos_;
        }
        while (pos_ < src_.size() &&
               ((src_[pos_] >= '0' && src_[pos_] <= '9') || src_[pos_] == '.' ||
                src_[pos_] == 'e' || src_[pos_] == 'E' || src_[pos_] == '+' || src_[pos_] == '-')) {
            ++pos_;
        }
        if (start == pos_) {
            return nullptr;
        }
        std::string tok = src_.substr(start, pos_ - start);
        char* end = nullptr;
        double num = std::strtod(tok.c_str(), &end);
        if (end == nullptr || *end != '\0') {
            return nullptr;
        }
        return std::make_unique<Value>(Value::Number(num));
    }

    std::unique_ptr<Value> ParseStringValue() {
        std::string out;
        if (!ConsumeString(out)) {
            return nullptr;
        }
        return std::make_unique<Value>(Value::String(std::move(out)));
    }

    bool ConsumeString(std::string& out) {
        if (pos_ >= src_.size() || src_[pos_] != '"') {
            return false;
        }
        ++pos_;
        while (pos_ < src_.size()) {
            char c = src_[pos_++];
            if (c == '"') {
                return true;
            }
            if (c == '\\') {
                if (pos_ >= src_.size()) {
                    return false;
                }
                char esc = src_[pos_++];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > src_.size()) {
                            return false;
                        }
                        std::string hex = src_.substr(pos_, 4);
                        pos_ += 4;
                        char* end = nullptr;
                        long code = std::strtol(hex.c_str(), &end, 16);
                        if (end == nullptr || *end != '\0') {
                            return false;
                        }
                        AppendUtf8(out, static_cast<std::uint32_t>(code));
                        break;
                    }
                    default:
                        return false;
                }
            } else {
                out.push_back(c);
            }
        }
        return false;  // unterminated
    }

    static void AppendUtf8(std::string& out, std::uint32_t code) {
        if (code < 0x80) {
            out.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else if (code < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }

    std::unique_ptr<Value> ParseObject() {
        ++pos_;  // '{'
        Value obj = Value::Object();
        SkipWhitespace();
        if (pos_ < src_.size() && src_[pos_] == '}') {
            ++pos_;
            return std::make_unique<Value>(std::move(obj));
        }
        while (pos_ < src_.size()) {
            SkipWhitespace();
            if (pos_ >= src_.size() || src_[pos_] != '"') {
                return nullptr;
            }
            std::string key;
            if (!ConsumeString(key)) {
                return nullptr;
            }
            SkipWhitespace();
            if (pos_ >= src_.size() || src_[pos_] != ':') {
                return nullptr;
            }
            ++pos_;
            SkipWhitespace();
            auto val = ParseValue();
            if (val == nullptr) {
                return nullptr;
            }
            obj.set(key, std::move(*val));
            SkipWhitespace();
            if (pos_ >= src_.size()) {
                return nullptr;
            }
            char c = src_[pos_++];
            if (c == ',') {
                continue;
            }
            if (c == '}') {
                return std::make_unique<Value>(std::move(obj));
            }
            return nullptr;
        }
        return nullptr;
    }

    std::unique_ptr<Value> ParseArray() {
        ++pos_;  // '['
        Value arr = Value::Array();
        SkipWhitespace();
        if (pos_ < src_.size() && src_[pos_] == ']') {
            ++pos_;
            return std::make_unique<Value>(std::move(arr));
        }
        while (pos_ < src_.size()) {
            SkipWhitespace();
            auto val = ParseValue();
            if (val == nullptr) {
                return nullptr;
            }
            arr.push(std::move(*val));
            SkipWhitespace();
            if (pos_ >= src_.size()) {
                return nullptr;
            }
            char c = src_[pos_++];
            if (c == ',') {
                continue;
            }
            if (c == ']') {
                return std::make_unique<Value>(std::move(arr));
            }
            return nullptr;
        }
        return nullptr;
    }

    void SkipWhitespace() {
        while (pos_ < src_.size() &&
               (src_[pos_] == ' ' || src_[pos_] == '\t' || src_[pos_] == '\n' || src_[pos_] == '\r')) {
            ++pos_;
        }
    }

    const std::string& src_;
    std::size_t pos_ = 0;
};

}  // namespace

std::unique_ptr<Value> Parse(const std::string& text) {
    Parser parser(text);
    return parser.ParseDocument();
}

namespace {

void AppendString(std::string& out, const std::string& s) {
    out.push_back('"');
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

void Append(std::string& out, const Value& v) {
    switch (v.type()) {
        case Type::Null: out += "null"; break;
        case Type::Bool: out += v.as_bool() ? "true" : "false"; break;
        case Type::Number: {
            double n = v.as_number();
            char buf[32];
            if (n == static_cast<int64_t>(n)) {
                std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(n));
            } else {
                std::snprintf(buf, sizeof(buf), "%g", n);
            }
            out += buf;
            break;
        }
        case Type::String: AppendString(out, v.as_string()); break;
        case Type::Object: {
            out.push_back('{');
            bool first = true;
            for (const auto& kv : v.members()) {
                if (!first) {
                    out.push_back(',');
                }
                first = false;
                AppendString(out, kv.first);
                out.push_back(':');
                Append(out, *kv.second);
            }
            out.push_back('}');
            break;
        }
        case Type::Array: {
            out.push_back('[');
            bool first = true;
            for (const auto& item : v.items()) {
                if (!first) {
                    out.push_back(',');
                }
                first = false;
                Append(out, *item);
            }
            out.push_back(']');
            break;
        }
    }
}

}  // namespace

std::string Serialize(const Value& value) {
    std::string out;
    Append(out, value);
    return out;
}

}  // namespace json
}  // namespace hllm
