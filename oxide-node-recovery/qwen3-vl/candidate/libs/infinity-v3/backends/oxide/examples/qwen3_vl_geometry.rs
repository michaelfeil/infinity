// Print the Qwen3-VL geometry coverage report (CPU only).
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = infinity_backend_oxide::qwen3_vl_geometry::report();
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
