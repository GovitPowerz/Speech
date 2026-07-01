//! Exact port of the legacy compile-time GFFT (V. Myrnyy, DDJ 2007).
//!
//! Ported from legacy C++: `fft.hpp`. Radix-2 decimation-in-time with (1) twiddle
//! SEEDS from a truncated Taylor series (`SinCosSeries`), not libm `sin`, and (2) a
//! Numerical-Recipes RUNNING twiddle recurrence that accumulates rounding error
//! across the butterfly loop. Bit-exactness against the legacy dumps requires
//! reproducing BOTH exactly, including operation order.
//!
//! In-place, interleaved `[re0, im0, re1, im1, ...]`, length `2 * (1 << p)`,
//! unnormalized, negative exponent. `GFFT<P>` transforms `N = 1 << p` complex
//! points. The engine registers `P` in `[1, 19]` (factory Min=1, Max=20 clamped to
//! Max-1). `Gfft::new` asserts that range, matching the legacy factory.

use std::f64::consts::PI;

/// Legacy `Sin<B,A,double>::value()` == `(A*PI/B) * SinCosSeries<2,34,B,A>::value()`.
///
/// The series is `S(M) = 1 - ((x*x)/M)/(M+1) * S(M+2)` with base `S(34) = 1`, and
/// `x = (A*PI)/B` (with `A*PI` evaluated first, matching the C++ `A*_PI/B`). The
/// C++ template is a compile-time constant; here it is a runtime recurrence with
/// identical f64 operation order, which reproduces gcc's strict-mode folding bit
/// for bit. `pub` so the seed test (an external integration-test crate) can
/// cross-check it against an independent coding.
pub fn sin_series(b: u64, a: u64) -> f64 {
    let x = (a as f64 * PI) / b as f64;
    // Recursive S(M,N) with M=2, N=34: recurse to the base then fold outward.
    fn s(m: u64, x: f64) -> f64 {
        if m == 34 {
            return 1.0;
        }
        1.0 - ((x * x) / m as f64) / (m as f64 + 1.0) * s(m + 2, x)
    }
    x * s(2, x)
}

/// Generic FFT object; `p` is the transform power (`N = 1 << p` complex points).
pub struct Gfft {
    p: u32,
}

impl Gfft {
    /// Build a GFFT for power `p`. Asserts `1 <= p <= 19` (legacy factory range).
    pub fn new(p: u32) -> Gfft {
        assert!(
            (1..=19).contains(&p),
            "GFFT power p={p} out of range [1, 19]"
        );
        Gfft { p }
    }

    /// In-place FFT: `data` is `2 * (1 << p)` interleaved re/im f64, negative
    /// exponent, unnormalized. Matches `fft.hpp` `GFFT::fft`: scramble then recurse.
    pub fn fft(&self, data: &mut [f64]) {
        let n = 1usize << self.p; // complex points
        assert_eq!(data.len(), 2 * n, "data len must be 2 * (1 << p)");
        scramble(data, n);
        danielson_lanczos(data, n);
    }
}

/// Numerical-Recipes bit-reversal (`fft.hpp:176-190`), ported from the 1-based int
/// loop verbatim (`i, j, m` as in the source; `data[j-1]`/`data[i-1]` etc.).
fn scramble(data: &mut [f64], n: usize) {
    let two_n = (2 * n) as i64;
    let n_i = n as i64;
    let mut j: i64 = 1;
    let mut i: i64 = 1;
    while i < two_n {
        if j > i {
            data.swap((j - 1) as usize, (i - 1) as usize);
            data.swap(j as usize, i as usize);
        }
        let mut m = n_i;
        while m >= 2 && j > m {
            j -= m;
            m >>= 1;
        }
        j += m;
        i += 2;
    }
}

/// `DanielsonLanczos<N>::apply` (`fft.hpp:78-148`). `n` is the template value (the
/// number of complex points of this block); the block spans `data[0..2*n]`. Split
/// at offset `n` doubles (each half = `n/2` complex = `n` doubles), then the
/// running-twiddle butterfly. Base cases `n==4` and `n==2` are the hard-coded
/// blocks (exact statement order preserved).
fn danielson_lanczos(data: &mut [f64], n: usize) {
    if n == 4 {
        // DanielsonLanczos<4> hard-coded fused -i butterfly (fft.hpp:107-136).
        let mut tr = data[2];
        let mut ti = data[3];
        data[2] = data[0] - tr;
        data[3] = data[1] - ti;
        data[0] += tr;
        data[1] += ti;
        tr = data[6];
        ti = data[7];
        data[6] = data[5] - ti;
        data[7] = tr - data[4];
        data[4] += tr;
        data[5] += ti;

        tr = data[4];
        ti = data[5];
        data[4] = data[0] - tr;
        data[5] = data[1] - ti;
        data[0] += tr;
        data[1] += ti;
        tr = data[6];
        ti = data[7];
        data[6] = data[2] - tr;
        data[7] = data[3] - ti;
        data[2] += tr;
        data[3] += ti;
        return;
    }
    if n == 2 {
        // DanielsonLanczos<2> plain butterfly (fft.hpp:138-148).
        let tr = data[2];
        let ti = data[3];
        data[2] = data[0] - tr;
        data[3] = data[1] - ti;
        data[0] += tr;
        data[1] += ti;
        return;
    }

    // General case (fft.hpp:78-103): recurse on both halves, then butterfly.
    danielson_lanczos(&mut data[..n], n / 2);
    danielson_lanczos(&mut data[n..], n / 2);

    // Twiddle seeds: wtemp = -Sin(N,1); wpr = -2*wtemp*wtemp; wpi = -Sin(N,2).
    let mut wtemp = -sin_series(n as u64, 1);
    let wpr = -2.0 * wtemp * wtemp;
    let wpi = -sin_series(n as u64, 2);
    let mut wr = 1.0_f64;
    let mut wi = 0.0_f64;

    let mut i = 0usize;
    while i < n {
        let tempr = data[i + n] * wr - data[i + n + 1] * wi;
        let tempi = data[i + n] * wi + data[i + n + 1] * wr;
        data[i + n] = data[i] - tempr;
        data[i + n + 1] = data[i + 1] - tempi;
        data[i] += tempr;
        data[i + 1] += tempi;

        wtemp = wr;
        wr += wr * wpr - wi * wpi;
        wi += wi * wpr + wtemp * wpi;
        i += 2;
    }
}
