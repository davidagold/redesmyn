use std::process::Command;

#[test]
fn demo_error_task_not_found_maps_to_exit_code_and_message() {
    let bin = env!("CARGO_BIN_EXE_redesmyn-server");
    let output = Command::new(bin)
        .args(["demo-error", "T-404"])
        .output()
        .expect("failed to run redesmyn-server");

    assert_eq!(output.status.code(), Some(3));

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error[not_found]"));
    assert!(stderr.contains("Task not found."));
}

#[test]
fn demo_error_invalid_task_id_maps_to_exit_code_and_message() {
    let bin = env!("CARGO_BIN_EXE_redesmyn-server");
    let output = Command::new(bin)
        // No task id argument -> empty string passed to demo -> invalid_request.
        .args(["demo-error"])
        .output()
        .expect("failed to run redesmyn-server");

    assert_eq!(output.status.code(), Some(2));

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error[invalid_request]"));
    assert!(stderr.contains("Invalid task id."));
}
