#pragma once

/*
 * Json.h — a tiny, dependency-free JSON parser used by the runtime's
 * package/manifest layer. It intentionally supports only the value shapes the
 * HLLM manifest and the CANN LLM Engine's context.json / executor.json need.
 * The device build may substitute a platform JSON library for this type
 * without touching the callers, as long as the accessor surface stays the same.
 */

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace hllm {
namespace json {

enum class Type {
    Null,
    Bool,
    Number,
    String,
    Object,
    Array,
};

class Value {
public:
    Value() = default;

    static Value Null() { return Value(Type::Null); }
    static Value Bool(bool b) {
        Value v(Type::Bool);
        v.bool_ = b;
        return v;
    }
    static Value Number(double n) {
        Value v(Type::Number);
        v.number_ = n;
        return v;
    }
    static Value String(std::string s) {
        Value v(Type::String);
        v.string_ = std::move(s);
        return v;
    }
    static Value Object() { return Value(Type::Object); }
    static Value Array() { return Value(Type::Array); }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_object() const { return type_ == Type::Object; }
    bool is_array() const { return type_ == Type::Array; }

    bool as_bool(bool fallback = false) const { return is_bool() ? bool_ : fallback; }
    double as_number(double fallback = 0.0) const { return is_number() ? number_ : fallback; }
    std::int64_t as_int64(std::int64_t fallback = 0) const {
        return is_number() ? static_cast<std::int64_t>(number_) : fallback;
    }
    const std::string& as_string() const {
        static const std::string empty;
        return is_string() ? string_ : empty;
    }

    // Object accessors.
    Value* get(const std::string& key) {
        if (!is_object()) {
            return nullptr;
        }
        auto it = object_.find(key);
        return it == object_.end() ? nullptr : it->second.get();
    }
    const Value* get(const std::string& key) const {
        if (!is_object()) {
            return nullptr;
        }
        auto it = object_.find(key);
        return it == object_.end() ? nullptr : it->second.get();
    }
    bool has(const std::string& key) const { return get(key) != nullptr; }

    void set(const std::string& key, Value v) {
        if (!is_object()) {
            type_ = Type::Object;
        }
        object_[key] = std::make_unique<Value>(std::move(v));
    }

    // Array accessors.
    Value* at(std::size_t i) {
        return i < array_.size() ? array_[i].get() : nullptr;
    }
    const Value* at(std::size_t i) const {
        return i < array_.size() ? array_[i].get() : nullptr;
    }
    std::size_t size() const { return array_.size(); }
    void push(Value v) { array_.push_back(std::make_unique<Value>(std::move(v))); }

    const std::vector<std::unique_ptr<Value>>& items() const { return array_; }
    const std::map<std::string, std::unique_ptr<Value>>& members() const { return object_; }

private:
    explicit Value(Type t) : type_(t) {}

    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::map<std::string, std::unique_ptr<Value>> object_;
    std::vector<std::unique_ptr<Value>> array_;
};

/*
 * Parse a complete JSON document. Returns nullptr on any syntax error. The
 * entire document must be consumed (no trailing junk).
 */
std::unique_ptr<Value> Parse(const std::string& text);

}  // namespace json
}  // namespace hllm
