//! KP専用の最小JSONパーサ（外部クレート不使用、ゼロから自作）。
//!
//! config.json / tokenizer_config.json / model.safetensors.index.json のような
//! 設定・メタデータの読み込みが目的。
//! - 整数は i64、小数・指数付き・i64に収まらないものは f64 で保持する。
//! - オブジェクトは出現順を保つ Vec で持つ（キー重複時 `get` は最初の一致を返す）。

use std::fmt;

const MAX_DEPTH: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub pos: usize,
    pub msg: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JSON parse error at byte {}: {}", self.pos, self.msg)
    }
}

impl std::error::Error for ParseError {}

impl Value {
    pub fn parse(src: &str) -> Result<Value, ParseError> {
        let mut p = Parser { src, b: src.as_bytes(), i: 0 };
        let v = p.value(0)?;
        p.ws();
        if p.i != p.b.len() {
            return p.fail("trailing characters after value");
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn at(&self, idx: usize) -> Option<&Value> {
        match self {
            Value::Array(a) => a.get(idx),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// 整数として読む。小数でも小数部が0で i64 に収まるなら許す（`1.0` など）。
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(n) => Some(*n),
            Value::Float(x) if x.fract() == 0.0 && x.abs() < 9.0e18 => Some(*x as i64),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_i64().and_then(|n| usize::try_from(n).ok())
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(n) => Some(*n as f64),
            Value::Float(x) => Some(*x),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Object(m) => Some(m),
            _ => None,
        }
    }
}

struct Parser<'a> {
    src: &'a str,
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn fail<T>(&self, msg: &str) -> Result<T, ParseError> {
        Err(ParseError { pos: self.i, msg: msg.to_string() })
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn expect(&mut self, c: u8) -> Result<(), ParseError> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            self.fail(&format!("expected '{}'", c as char))
        }
    }

    fn literal(&mut self, word: &str, v: Value) -> Result<Value, ParseError> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            self.fail("invalid literal")
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, ParseError> {
        if depth > MAX_DEPTH {
            return self.fail("nesting too deep");
        }
        self.ws();
        match self.peek() {
            None => self.fail("unexpected end of input"),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => self.fail("unexpected character"),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.i += 1; // '{'
        let mut members = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Value::Object(members));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return self.fail("expected string key");
            }
            let key = self.string()?;
            self.ws();
            self.expect(b':')?;
            let v = self.value(depth + 1)?;
            members.push((key, v));
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::Object(members));
                }
                _ => return self.fail("expected ',' or '}'"),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.i += 1; // '['
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::Array(items));
                }
                _ => return self.fail("expected ',' or ']'"),
            }
        }
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.i += 1; // 開き引用符
        let mut out = String::new();
        loop {
            let start = self.i;
            while let Some(c) = self.peek() {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.i += 1;
            }
            // 区切りは全てASCIIなので、start..i は常に文字境界。
            out.push_str(&self.src[start..self.i]);
            match self.peek() {
                None => return self.fail("unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    self.escape(&mut out)?;
                }
                Some(_) => return self.fail("control character in string"),
            }
        }
    }

    fn escape(&mut self, out: &mut String) -> Result<(), ParseError> {
        let c = match self.peek() {
            Some(c) => c,
            None => return self.fail("unterminated escape"),
        };
        self.i += 1;
        match c {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{8}'),
            b'f' => out.push('\u{c}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                let ch = self.unicode()?;
                out.push(ch);
            }
            _ => {
                self.i -= 1;
                return self.fail("invalid escape");
            }
        }
        Ok(())
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        if self.i + 4 > self.b.len() {
            return self.fail("truncated \\u escape");
        }
        let mut v = 0u32;
        for k in 0..4 {
            let d = match self.b[self.i + k] {
                c @ b'0'..=b'9' => c - b'0',
                c @ b'a'..=b'f' => c - b'a' + 10,
                c @ b'A'..=b'F' => c - b'A' + 10,
                _ => return self.fail("invalid hex digit in \\u escape"),
            };
            v = v * 16 + d as u32;
        }
        self.i += 4;
        Ok(v)
    }

    fn unicode(&mut self) -> Result<char, ParseError> {
        let hi = self.hex4()?;
        let cp = match hi {
            0xD800..=0xDBFF => {
                if self.peek() == Some(b'\\') && self.b.get(self.i + 1) == Some(&b'u') {
                    self.i += 2;
                    let lo = self.hex4()?;
                    if !(0xDC00..=0xDFFF).contains(&lo) {
                        return self.fail("invalid low surrogate");
                    }
                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                } else {
                    return self.fail("lone high surrogate");
                }
            }
            0xDC00..=0xDFFF => return self.fail("lone low surrogate"),
            _ => hi,
        };
        match char::from_u32(cp) {
            Some(c) => Ok(c),
            None => self.fail("invalid code point"),
        }
    }

    fn digits(&mut self) -> usize {
        let s = self.i;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        self.i - s
    }

    fn number(&mut self) -> Result<Value, ParseError> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.i += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return self.fail("leading zero in number");
                }
            }
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return self.fail("invalid number"),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.i += 1;
            if self.digits() == 0 {
                return self.fail("expected digits after '.'");
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.digits() == 0 {
                return self.fail("expected digits in exponent");
            }
        }
        let text = &self.src[start..self.i];
        if !is_float {
            if let Ok(n) = text.parse::<i64>() {
                return Ok(Value::Int(n));
            }
        }
        match text.parse::<f64>() {
            Ok(x) => Ok(Value::Float(x)),
            Err(_) => self.fail("invalid number"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Value {
        Value::parse(s).unwrap_or_else(|e| panic!("{s:?}: {e}"))
    }

    fn err(s: &str) -> ParseError {
        Value::parse(s).expect_err(s)
    }

    #[test]
    fn scalars() {
        assert_eq!(p("null"), Value::Null);
        assert_eq!(p(" true "), Value::Bool(true));
        assert_eq!(p("false"), Value::Bool(false));
        assert_eq!(p("0"), Value::Int(0));
        assert_eq!(p("-0"), Value::Int(0));
        assert_eq!(p("-12"), Value::Int(-12));
        assert_eq!(p("1.5"), Value::Float(1.5));
        assert_eq!(p("1e3"), Value::Float(1000.0));
        assert_eq!(p("2.5E-2"), Value::Float(0.025));
        assert_eq!(p("1e-05"), Value::Float(1e-5));
    }

    #[test]
    fn big_integers_fall_back_to_float() {
        assert_eq!(p("9223372036854775807"), Value::Int(i64::MAX));
        match p("9223372036854775808") {
            Value::Float(x) => assert!((x - 9.223372036854775808e18).abs() < 1.0),
            v => panic!("{v:?}"),
        }
    }

    #[test]
    fn strings_and_escapes() {
        assert_eq!(p(r#""abc""#), Value::Str("abc".into()));
        assert_eq!(p(r#""a\"b\\c\/d""#), Value::Str("a\"b\\c/d".into()));
        assert_eq!(p(r#""\b\f\n\r\t""#), Value::Str("\u{8}\u{c}\n\r\t".into()));
        assert_eq!(p(r#""\u0041\u00e9""#), Value::Str("Aé".into()));
        assert_eq!(p("\"日本語🙂\""), Value::Str("日本語🙂".into()));
        // サロゲートペア: U+1F642
        assert_eq!(p(r#""\ud83d\ude42""#), Value::Str("🙂".into()));
    }

    #[test]
    fn objects_keep_order() {
        let v = p(r#"{"b": 1, "a": [true, null, {"x": "y"}], "c": {}}"#);
        let keys: Vec<&str> = v.as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["b", "a", "c"]);
        assert_eq!(v.get("b").and_then(Value::as_i64), Some(1));
        assert_eq!(v.get("a").and_then(|a| a.at(2)).and_then(|o| o.get("x")).and_then(Value::as_str), Some("y"));
        assert_eq!(v.get("c").and_then(Value::as_object).map(|m| m.len()), Some(0));
        assert!(v.get("missing").is_none());
    }

    #[test]
    fn accessors() {
        assert_eq!(p("3.0").as_i64(), Some(3));
        assert_eq!(p("3.5").as_i64(), None);
        assert_eq!(p("-1").as_usize(), None);
        assert_eq!(p("7").as_f64(), Some(7.0));
        assert_eq!(p("[]").as_array().map(|a| a.len()), Some(0));
        assert!(p("null").is_null());
    }

    #[test]
    fn rejects_malformed() {
        assert!(err("").msg.contains("end of input"));
        assert!(err("[1,]").msg.contains("unexpected"));
        assert!(err("{\"a\":1,}").msg.contains("string key"));
        assert!(err("{\"a\" 1}").msg.contains(":"));
        assert!(err("01").msg.contains("leading zero"));
        assert!(err("1.").msg.contains("digits"));
        assert!(err("1e").msg.contains("exponent"));
        assert!(err("-").msg.contains("invalid number"));
        assert!(err("\"abc").msg.contains("unterminated"));
        assert!(err("\"a\nb\"").msg.contains("control"));
        assert!(err(r#""\x""#).msg.contains("escape"));
        assert!(err(r#""\ud83d""#).msg.contains("surrogate"));
        assert!(err(r#""\ude42""#).msg.contains("surrogate"));
        assert!(err(r#""\u12""#).msg.contains("\\u"));
        assert!(err("nul").msg.contains("literal"));
        assert!(err("1 2").msg.contains("trailing"));
    }

    #[test]
    fn depth_limit() {
        let deep = "[".repeat(MAX_DEPTH + 10);
        assert!(err(&deep).msg.contains("too deep"));
        let ok = format!("{}{}", "[".repeat(100), "]".repeat(100));
        assert!(Value::parse(&ok).is_ok());
    }
}
