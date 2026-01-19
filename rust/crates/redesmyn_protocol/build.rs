use std::path::{Path, PathBuf};

fn proto_root(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join("../../proto")
}

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR missing"));
    let proto_root = proto_root(&manifest_dir);

    let proto_files = [
        "envelope.proto",
        "daemon.proto",
        "client.proto",
        "artifacts.proto",
    ];

    for proto in proto_files {
        println!(
            "cargo:rerun-if-changed={}",
            proto_root.join(proto).display()
        );
    }

    let protoc_path =
        protoc_bin_vendored::protoc_bin_path().expect("failed to locate vendored protoc binary");
    let protoc_include = protoc_bin_vendored::include_path()
        .expect("failed to locate vendored protoc include directory");

    let protos: Vec<PathBuf> = proto_files
        .iter()
        .map(|name| proto_root.join(name))
        .collect();

    let mut config = prost_build::Config::new();
    config.protoc_executable(protoc_path);
    config.include_file("redesmyn_protocol_pb.rs");

    config
        .compile_protos(&protos, &[proto_root, protoc_include])
        .expect("failed to compile protobuf schemas");
}
