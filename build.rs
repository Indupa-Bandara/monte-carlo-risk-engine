#[cfg(windows)]
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    configure_cmake_executable();
    let destination = cmake::Config::new("cpp").build();
    println!(
        "cargo:rustc-link-search=native={}",
        destination.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=risk_engine_cpp");
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => println!("cargo:rustc-link-lib=dylib=c++"),
        Ok("windows") => {}
        Ok(_) => println!("cargo:rustc-link-lib=dylib=stdc++"),
        Err(error) => panic!("missing CARGO_CFG_TARGET_OS: {error}"),
    }
    println!("cargo:rerun-if-changed=cpp/CMakeLists.txt");
    println!("cargo:rerun-if-changed=cpp/include/risk_engine.h");
    println!("cargo:rerun-if-changed=cpp/src/risk_engine.cpp");
    println!("cargo:rerun-if-changed=cpp/src/risk_engine_avx2.cpp");
    println!("cargo:rerun-if-changed=cpp/src/kernel_common.h");
    println!("cargo:rerun-if-env-changed=CMAKE");
}

fn configure_cmake_executable() {
    if std::env::var_os("CMAKE").is_some()
        || Command::new("cmake")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    {
        return;
    }

    #[cfg(windows)]
    if let Some(cmake) = find_visual_studio_cmake() {
        std::env::set_var("CMAKE", cmake);
    }
}

#[cfg(windows)]
fn find_visual_studio_cmake() -> Option<PathBuf> {
    let roots = [
        std::env::var_os("ProgramFiles").map(PathBuf::from),
        std::env::var_os("ProgramFiles(x86)").map(PathBuf::from),
    ];
    for root in roots.into_iter().flatten() {
        let installations = root.join("Microsoft Visual Studio");
        for version in read_directories(&installations) {
            for edition in read_directories(&version) {
                let cmake = edition.join(
                    "Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe",
                );
                if cmake.is_file() {
                    return Some(cmake);
                }
            }
        }
    }
    None
}

#[cfg(windows)]
fn read_directories(path: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect()
}
