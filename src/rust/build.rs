// Build provenance for `speech::build_info()`: the target triple, the Cargo
// profile and the rustc version, baked in at compile time so a ledger record
// (issue #20) can name the build that produced a number. `TARGET` and
// `PROFILE` are only visible to build scripts, hence the re-export.
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    let target = std::env::var("TARGET").unwrap_or_default();
    let profile = std::env::var("PROFILE").unwrap_or_default();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=SPEECH_BUILD_TARGET={target}");
    println!("cargo:rustc-env=SPEECH_BUILD_PROFILE={profile}");
    println!("cargo:rustc-env=SPEECH_BUILD_RUSTC={version}");
}
