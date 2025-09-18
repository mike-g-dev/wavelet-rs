use std::time::Duration;
use wavelet::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TODO: spinning works but sleep does not
    let mut runtime = Runtime::builder()
        .with_clock(PrecisionClock::new())
        .with_mode(Sleep::new(Duration::from_millis(100)))
        .build()?;

    // Create a data source
    let source = NodeBuilder::new(0u64)
        .on_init(|executor, _, idx| {
            executor.yield_driver().yield_now(idx);
        })
        .build(runtime.executor(), |counter, _ctx| {
            *counter += 1;
            println!("Source: {}", counter);
            Control::Broadcast
        });

    // Create a processor that reacts to the source
    let _processor = NodeBuilder::new(String::new()).triggered_by(&source).build(
        runtime.executor(),
        move |state, _ctx| {
            *state = format!("Processed: {}", source.borrow());
            println!("{}", state);
            Control::Unchanged
        },
    );

    // Run the graph
    runtime.run_forever();
    Ok(())
}
