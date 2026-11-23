use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mg_procedure_path = manifest_dir.join("mgp/mg_procedure.h");
    let mg_procedure_str = mg_procedure_path
        .to_str()
        .expect("Invalid mg_procedure.h path");

    println!("cargo:rerun-if-changed={}", mg_procedure_str);

    let bindings = bindgen::Builder::default()
        .header(mg_procedure_str)
        .blocklist_function("mgp_*")
        .rustified_enum("mgp_error")
        .rustified_enum("mgp_value_type")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings")
        .to_string();

    let mut bindings_string = "#[cfg_attr(test, mockall::automock)]\npub mod ffi {\nuse super::*;"
        .to_owned();
    bindings_string.push_str(
        &bindgen::Builder::default()
            .header(mg_procedure_str)
            .blocklist_type(".*")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings")
            .to_string(),
    );
    bindings_string.push_str("}\n");
    bindings_string.push_str(&bindings);

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    fs::write(out_path, bindings_string.as_bytes()).expect("Couldn't write bindings!");
}
