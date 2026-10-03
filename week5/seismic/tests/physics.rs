use std::process::Command;
#[test]
fn enzyme_cube_and_step_derivatives() {
    let output = Command::new(env!("CARGO_BIN_EXE_seismic"))
        .arg("--self-test")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("Enzyme cube and timestep PASS")
    );
}
