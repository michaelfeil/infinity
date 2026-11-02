#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(error) = q35moe_unified_runtime::cli::run(&args).await {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
