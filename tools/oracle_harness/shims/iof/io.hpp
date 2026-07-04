#pragma once
// Faithful stand-in for the vendored "iof" formatting library (absent from
// legacy/src/). Phase 2b links the real Segmentation::toFile_VRCTS as a byte
// golden, so the insertion path must format exactly like the disassembled iof:
//   %f.Ns == std::fixed << std::setprecision(N)   (N observed in {0,2,3,4})
//   %s    == default stream insertion
//   %%    == a literal '%'
// Unknown / width-flag directives (% 6ds, %f3.2s, % f6.2s, %s.s, ...) only occur
// on log paths the harness never dumps as a golden; we emit their literal prefix
// then treat the directive as %s (documented simplification -- they never reach a
// golden, so no width/flag fidelity is required).
//
// Usage patterns in the linked TUs:
//   os  << iof::fmtr("...") << v0 << v1 ...   (insertion; the golden path)
//   iss >> iof::fmtr("...") >> v0 ...         (extraction; parse paths, never a
//                                              golden -- kept as inert passthrough)
#include <iomanip>
#include <istream>
#include <ostream>
#include <string>

namespace iof {

struct fmtr {
    std::string f;
    explicit fmtr(const char* s) : f(s) {}
    explicit fmtr(const std::string& s) : f(s) {}
};

// Proxy returned by `ostream << fmtr`. Holds the target stream, the format
// string, and a cursor into it. Each chained `<< value` emits the literal run up
// to the next `%` directive (expanding `%%` to a literal `%`), then formats the
// value per the directive. Once the format string is exhausted, the trailing
// literal is flushed on the first spare insert and further inserts pass through
// raw (the tail-flush + raw-passthrough contract, e.g. Segmenter.cpp:66-67).
class fmtr_proxy {
public:
    fmtr_proxy(std::ostream& os, const std::string& f) : os_(os), f_(f), pos_(0) {}

    template <typename T>
    fmtr_proxy& operator<<(const T& value) {
        if (pos_ >= f_.size()) {
            os_ << value;  // format exhausted: raw passthrough
            return *this;
        }
        emit_literal_to_directive();
        if (pos_ >= f_.size()) {
            os_ << value;  // no directive left after the literal: raw passthrough
            return *this;
        }
        apply_directive(value);
        // If that was the last directive, flush the remaining literal tail now so
        // callers that stop chaining still see it.
        flush_if_tail_only();
        return *this;
    }

private:
    // Emit literal text starting at pos_ up to (not including) the next real
    // directive; expand `%%` to a single `%`. Leaves pos_ at the `%` of the
    // directive, or at f_.size() if none remains.
    void emit_literal_to_directive() {
        while (pos_ < f_.size()) {
            char c = f_[pos_];
            if (c != '%') {
                os_ << c;
                ++pos_;
                continue;
            }
            if (pos_ + 1 < f_.size() && f_[pos_ + 1] == '%') {
                os_ << '%';  // "%%" -> literal '%'
                pos_ += 2;
                continue;
            }
            return;  // pos_ sits on a directive-opening '%'
        }
    }

    // Consume the directive at pos_ and insert `value` formatted accordingly.
    // `%f.Ns` -> fixed/setprecision(N); anything else -> default insertion (%s
    // and all unknown/width forms). Advances pos_ past the directive's trailing
    // 's' (directives in the linked TUs all terminate at 's').
    template <typename T>
    void apply_directive(const T& value) {
        // pos_ points at '%'. Scan the raw directive body up to and including the
        // terminating 's'.
        std::string::size_type i = pos_ + 1;
        int precision = -1;
        bool fixed_prec = false;
        // Exactly "%f.Ns": 'f', '.', digits, 's' with nothing else in between.
        if (i < f_.size() && f_[i] == 'f' && i + 1 < f_.size() && f_[i + 1] == '.') {
            std::string::size_type j = i + 2;
            int n = 0;
            bool got_digit = false;
            while (j < f_.size() && f_[j] >= '0' && f_[j] <= '9') {
                n = n * 10 + (f_[j] - '0');
                got_digit = true;
                ++j;
            }
            if (got_digit && j < f_.size() && f_[j] == 's') {
                fixed_prec = true;
                precision = n;
                i = j;  // 's'
            }
        }
        // Advance i to the directive's terminating 's' (covers %s and every
        // unknown/width form, which all end at 's').
        while (i < f_.size() && f_[i] != 's') ++i;
        if (fixed_prec) {
            std::ios_base::fmtflags flags = os_.flags();
            std::streamsize prec = os_.precision();
            os_ << std::fixed << std::setprecision(precision) << value;
            os_.flags(flags);
            os_.precision(prec);
        } else {
            os_ << value;  // %s and unknown/width directives
        }
        pos_ = (i < f_.size()) ? i + 1 : f_.size();  // past the 's'
    }

    // After consuming a directive, if no further directive remains, flush the
    // trailing literal immediately (expanding `%%`). Any later `<< value` then
    // passes through raw.
    void flush_if_tail_only() {
        std::string::size_type scan = pos_;
        while (scan < f_.size()) {
            if (f_[scan] == '%' && !(scan + 1 < f_.size() && f_[scan + 1] == '%')) {
                return;  // another directive remains; do not flush yet
            }
            scan += (f_[scan] == '%') ? 2 : 1;
        }
        emit_literal_to_directive();  // no directive left -> flush the tail
    }

    std::ostream& os_;
    const std::string f_;
    std::string::size_type pos_;
};

inline fmtr_proxy operator<<(std::ostream& os, const fmtr& fm) {
    return fmtr_proxy(os, fm.f);
}

// Extraction path: parse-only, never a golden. Kept inert (returns the stream
// unchanged) so chained `>> value` reads flow straight to the real operators.
inline std::istream& operator>>(std::istream& is, const fmtr&) { return is; }

}  // namespace iof
