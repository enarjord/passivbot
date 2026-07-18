fn main() {
    println!("cargo:rerun-if-env-changed=PASSIVBOT_RUST_SOURCE_FINGERPRINT");
}
