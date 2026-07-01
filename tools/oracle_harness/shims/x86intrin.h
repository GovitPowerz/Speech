#pragma once
// legacy fmath.hpp does #include <x86intrin.h> unconditionally on GCC/non-Windows.
// Apple Silicon (aarch64) has no x86 intrinsic headers, so we redirect to
// sse2neon (brew: sse2neon), a header-only translator of Intel SSE intrinsics to
// Arm NEON. fmath's SSE fast-exp/log routines are never odr-used by any harness
// translation unit (MelFilterBank uses std::exp/std::log, not fmath::), so this
// only needs to make fmath.hpp *parse*; sse2neon keeps it numerically faithful
// should a later task exercise those paths.
#include <sse2neon.h>
