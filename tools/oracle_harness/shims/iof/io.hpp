#pragma once
// Inert stand-in for the vendored "iof" formatting library, which is absent
// from legacy/src/. Every iof usage in the compiled translation units sits in
// logging / parsing paths the oracle harness never calls, so this shim only has
// to make the two syntactic patterns compile:
//   os  << iof::fmtr("...") << value ...   (insertion into an ostream)
//   iss >> iof::fmtr("...") >> value ...   (extraction from an istream)
// Both operators return the underlying stream unchanged, so any values chained
// after the fmtr flow straight through to the real stream operators.
#include <istream>
#include <ostream>
#include <string>

namespace iof {
struct fmtr {
    std::string f;
    explicit fmtr(const char* s) : f(s) {}
};
inline std::ostream& operator<<(std::ostream& os, const fmtr&) { return os; }
inline std::istream& operator>>(std::istream& is, const fmtr&) { return is; }
}  // namespace iof
