fn main() {
    // Forward the native bridge location to the server's linker.
    if let Ok(path) = std::env::var("DEP_INFINITY_TORCH_LIB_DIR") {
        println!("cargo:torch_lib_dir={path}");
    }
}
