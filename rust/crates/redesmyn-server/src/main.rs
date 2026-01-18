use redesmyn_protocol::ErrorEnvelope;

fn main() {
    redesmyn_logging::init();

    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("demo-error") {
        return;
    }

    // `redesmyn-server demo-error <task-id>` is a tiny end-to-end example for the
    // workspace error conventions (T-3). It will be replaced by real server
    // command/flag handling in later tickets.
    let task_id = args.next().unwrap_or_default();
    match redesmyn_control_plane::demo::get_task(&task_id) {
        Ok(task) => {
            println!("ok: {}", task.task_id);
        }
        Err(err) => {
            let envelope: ErrorEnvelope = err.into();
            eprintln!("error[{}]: {}", envelope.category, envelope.message);
            std::process::exit(envelope.exit_code());
        }
    }
}
