fn main() {
    redesmyn_logging::init();
    if let Err(err) = redesmyn_config::load_rust_config(Default::default()) {
        eprintln!("{err}");
        std::process::exit(2);
    }
}
