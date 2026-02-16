use crate::benchmark::ToBenchmarkFunction;
use crate::trace::{prepare_function_args, Function as TraceFunction};
use crate::utils::*;
use anyhow::{Context, Result};
use clap::Parser;
use iluvatar_controller_library::server::controller_comm::ControllerAPIFactory;
use iluvatar_library::clock::get_global_clock;
use iluvatar_library::tokio_utils::{build_tokio_runtime, TokioRuntime};
use iluvatar_library::types::Compute;
use iluvatar_library::utils::config::args_to_json;
use iluvatar_library::{transaction::gen_tid, utils::port_utils::Port};
use iluvatar_worker_library::worker_api::worker_comm::WorkerAPIFactory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
/// Parallel throughput test — spawns N threads (copies) per function.
/// Each thread deploys exactly one function name and then: invoke → sleep(w) → repeat.
pub struct ThroughputArgs {
    #[arg(short, long, value_enum)]
    /// Target for the load (Controller or Worker)
    target: Target,

    #[arg(long)]
    /// CSV with all functions to test (same schema as benchmark.rs)
    function_file: String,

    #[arg(long, default_value = "60")]
    /// Duration in seconds each thread runs for
    runtime_sec: u64,

    #[arg(long, default_value = "5")]
    /// Wait time 'w' in milliseconds between invocations *within a thread*
    wait_time_ms: u64,

    #[arg(short, long)]
    /// Port controller/worker is listening on
    port: Port,

    #[arg(long)]
    /// Host controller/worker is on
    host: String,

    #[arg(short, long)]
    /// Folder to output results to
    pub out_folder: String,

    #[arg(long, default_value = "1")]
    /// Number of *copies* (threads) per unique function
    copies_per_function: usize,

    #[arg(long)]
    /// Optional preferred compute (only for Worker). If a function has multiple computes,
    /// we will pick this one; if not present, we fall back to the first set bit.
    preferred_compute: Option<Compute>,

    #[arg(long, default_value_t = false)]
    /// Output logs to stdout (hook this into your global logger if desired)
    pub log_stdout: bool,
}

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct ThroughputResult {
    pub function_name: String,
    pub compute: Compute,
    pub total_invocations: usize,
    pub successful_invocations: usize,
    pub failed_invocations: usize,
    pub throughput_rps: f64, // success_rps
    pub offered_rps: f64,    // total/elapsed
}

pub fn throughput_functions(args: ThroughputArgs) -> Result<()> {
    let functions = load_functions_from_csv(&args.function_file)?;
    let threaded_rt = build_tokio_runtime(&None, &None, &None, &gen_tid())?;

    match args.target {
        Target::Worker => throughput_worker_parallel(&threaded_rt, functions, args),
        Target::Controller => threaded_rt.block_on(throughput_controller_parallel(functions, args)),
    }
}

/// Load functions from CSV, same schema as benchmark.rs
fn load_functions_from_csv(path: &str) -> Result<Vec<ToBenchmarkFunction>> {
    let mut functions = Vec::new();
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("Unable to open metadata csv file '{}'", path))?;
    for result in rdr.deserialize() {
        let func: ToBenchmarkFunction =
            result.with_context(|| "Error deserializing ToBenchmarkFunction from CSV row")?;
        functions.push(func);
    }
    Ok(functions)
}

/* ========================= Controller path (parallel) ========================= */

async fn throughput_controller_parallel(
    functions: Vec<ToBenchmarkFunction>,
    args: ThroughputArgs,
) -> Result<()> {
    let factory = ControllerAPIFactory::boxed();
    let clock = get_global_clock(&gen_tid())?;
    let mut handles = Vec::new();

    for f in &functions {
        for copy_id in 0..args.copies_per_function {
            let f = f.clone();
            let host = args.host.clone();
            let port = args.port;
            let factory = factory.clone();
            let clock = clock.clone();
            let runtime = args.runtime_sec;
            let wait_ms = args.wait_time_ms;

            // Choose a single compute flag to register with controller (from bitflags).
            // If your CSV encodes multiple computes in f.compute, we pick preferred or first bit.
            let chosen_compute = pick_compute_bitflag(f.compute, args.preferred_compute);

            handles.push(tokio::spawn(async move {
                let version = "0.0.1";
                // Unique name per copy to avoid registration/name clashes
                let base_name = format!("{}-throughput", f.name);
                let name = format!("{}-copy{}", base_name, copy_id);

                // Register once
                let api = factory.get_controller_api(&host, port, &gen_tid()).await?;
                if let Err(e) = controller_register(
                    &name,
                    version,
                    &f.image_name,
                    f.memory.unwrap_or(512),
                    f.isolation,
                    chosen_compute, // single flag
                    f.server,
                    None,
                    api.clone(),
                )
                    .await
                {
                    error!("Controller register failed for {}: {}", name, e);
                    anyhow::bail!(e);
                }

                // Invoke loop
                let start = Instant::now();
                let mut total = 0usize;
                let mut success = 0usize;
                while start.elapsed() < Duration::from_secs(runtime) {
                    match controller_invoke(&name, version, None, clock.clone(), api.clone()).await {
                        Ok(inv) => {
                            total += 1;
                            if inv.controller_response.success {
                                success += 1;
                            }
                        }
                        Err(e) => error!("[{}] invoke error: {}", name, e),
                    }
                    sleep(Duration::from_millis(wait_ms)).await;
                }

                let elapsed = start.elapsed().as_secs_f64().max(1e-9);
                let success_rps = success as f64 / elapsed;
                let offered_rps = total as f64 / elapsed;

                Ok::<(String, Compute, usize, usize, usize, f64, f64), anyhow::Error>((
                    f.name.clone(),
                    chosen_compute,
                    total,
                    success,
                    total - success,
                    success_rps,
                    offered_rps,
                ))
            }));
        }
    }

    // Gather and aggregate per function without futures::join_all
    let mut agg: HashMap<(String, Compute), ThroughputResult> = HashMap::new();
    for h in handles {
        match h.await {
            Ok(Ok((fname, comp, total, succ, fail, succ_rps, off_rps))) => {
                let e = agg.entry((fname.clone(), comp)).or_insert(ThroughputResult {
                    function_name: fname,
                    compute: comp,
                    total_invocations: 0,
                    successful_invocations: 0,
                    failed_invocations: 0,
                    throughput_rps: 0.0,
                    offered_rps: 0.0,
                });
                e.total_invocations += total;
                e.successful_invocations += succ;
                e.failed_invocations += fail;
                e.throughput_rps += succ_rps;
                e.offered_rps += off_rps;
            }
            Ok(Err(e)) => error!("Task failed: {:?}", e),
            Err(join_err) => error!("Join error: {:?}", join_err),
        }
    }

    let out: Vec<ThroughputResult> = agg.into_values().collect();
    let p = Path::new(&args.out_folder).join("controller_throughput_results.json");
    save_result_json(p, &out)?;
    info!("Controller throughput complete: {} function entries", out.len());
    Ok(())
}

/* =========================== Worker path (parallel) =========================== */

fn throughput_worker_parallel(
    threaded_rt: &TokioRuntime,
    functions: Vec<ToBenchmarkFunction>,
    args: ThroughputArgs,
) -> Result<()> {
    threaded_rt.block_on(async move {
        let factory = WorkerAPIFactory::boxed();
        let clock = get_global_clock(&gen_tid())?;

        let mut handles = Vec::new();

        for f in &functions {
            // Choose one compute flag per function (from bitflags).
            let compute = pick_compute_bitflag(f.compute, args.preferred_compute);

            // If multiple computes are present and no preference is given, we log which one we picked.
            if !is_single_bit(f.compute) && args.preferred_compute.is_none() {
                warn!(
                    "Function '{}' has multiple computes set; using {:?}. \
                     Pass --preferred-compute to control this.",
                    f.name, compute
                );
            }

            for copy_id in 0..args.copies_per_function {
                let f = f.clone();
                let factory = factory.clone();
                let clock = clock.clone();

                let host = args.host.clone();
                let port = args.port;
                let compute = compute; // chosen above
                let runtime = args.runtime_sec;
                let wait_ms = args.wait_time_ms;

                handles.push(tokio::spawn(async move {
                    let version = "0.0.1";
                    // Unique per-copy name to avoid collision
                    let reg_name = format!("{}.copy{}", f.name, copy_id);

                    // Build args (if any)
                    let mut dummy = TraceFunction::default();
                    let func_args = if let Some(arg) = &f.args {
                        dummy.args = Some(arg.to_string());
                        args_to_json(&prepare_function_args(&dummy, LoadType::Functions))?
                    } else {
                        "{\"name\":\"THROUGHPUT\"}".to_string()
                    };

                    // Register once
                    if let Err(e) = worker_register(
                        reg_name.clone(),
                        version,
                        f.image_name.clone(),
                        f.memory.unwrap_or(512),
                        host.clone(),
                        port,
                        &factory,
                        f.isolation,
                        compute,
                        f.server,
                        None,
                    )
                        .await
                    {
                        error!("Worker register failed for {}: {}", reg_name, e);
                        anyhow::bail!(e);
                    }

                    // Invoke loop
                    let start = Instant::now();
                    let mut total = 0usize;
                    let mut success = 0usize;
                    while start.elapsed() < Duration::from_secs(runtime) {
                        match worker_invoke(
                            &reg_name,
                            version,
                            &host,
                            port,
                            &gen_tid(),
                            Some(func_args.clone()),
                            clock.clone(),
                            &factory,
                        )
                            .await
                        {
                            Ok(resp) => {
                                total += 1;
                                if resp.worker_response.success {
                                    success += 1;
                                }
                            }
                            Err(e) => error!("[{}] invoke error: {}", reg_name, e),
                        }
                        sleep(Duration::from_millis(wait_ms)).await;
                    }

                    // Clean up only for non-CPU resources
                    if compute != Compute::CPU {
                        if let Err(e) = worker_clean(&host, port, &gen_tid(), &factory).await {
                            error!("[{}] worker clean failed: {:?}", reg_name, e);
                        }
                    }

                    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
                    let success_rps = success as f64 / elapsed;
                    let offered_rps = total as f64 / elapsed;

                    Ok::<(String, Compute, usize, usize, usize, f64, f64), anyhow::Error>((
                        f.name.clone(),
                        compute,
                        total,
                        success,
                        total - success,
                        success_rps,
                        offered_rps,
                    ))
                }));
            }
        }

        // Gather + aggregate per (function, compute)
        let mut agg: HashMap<(String, Compute), ThroughputResult> = HashMap::new();
        for h in handles {
            match h.await {
                Ok(Ok((fname, comp, total, succ, fail, succ_rps, off_rps))) => {
                    let e = agg.entry((fname.clone(), comp)).or_insert(ThroughputResult {
                        function_name: fname,
                        compute: comp,
                        total_invocations: 0,
                        successful_invocations: 0,
                        failed_invocations: 0,
                        throughput_rps: 0.0,
                        offered_rps: 0.0,
                    });
                    e.total_invocations += total;
                    e.successful_invocations += succ;
                    e.failed_invocations += fail;
                    e.throughput_rps += succ_rps;
                    e.offered_rps += off_rps;
                }
                Ok(Err(e)) => error!("Task failed: {:?}", e),
                Err(join_err) => error!("Join error: {:?}", join_err),
            }
        }

        // Write results
        let out: Vec<ThroughputResult> = agg.into_values().collect();
        let json_path = Path::new(&args.out_folder).join("worker_throughput_results.json");
        save_result_json(json_path, &out)?;
        info!(
            "Worker throughput complete: {} function entries (aggregated across copies)",
            out.len()
        );
        Ok(())
    })
}

/* ============================== Helpers ===================================== */

/// Pick a single `Compute` flag from a bitflag set.
/// Priority:
///   1) If --preferred-compute is set and contained in `available`, use it.
///   2) Else pick the **lowest set bit** from `available`.
///   3) If `available` is empty, fall back to `Compute::CPU`.
fn pick_compute_bitflag(available: Compute, preferred: Option<Compute>) -> Compute {
    if let Some(p) = preferred {
        if available.contains(p) {
            return p;
        }
        warn!(
            "Preferred compute {:?} not present in available {:?}; picking first set bit.",
            p, available
        );
    }
    if available.is_empty() {
        return Compute::CPU;
    }
    first_set_bit(available).unwrap_or(Compute::CPU)
}

/// Return the lowest set bit as a `Compute` flag.
/// Works for arbitrary bitflags.
fn first_set_bit(flags: Compute) -> Option<Compute> {
    let bits = flags.bits();
    if bits == 0 {
        return None;
    }
    let lsb = bits & bits.wrapping_neg();
    Compute::from_bits(lsb)
}

/// True if exactly one bit is set.
fn is_single_bit(flags: Compute) -> bool {
    let b = flags.bits();
    b != 0 && (b & (b - 1)) == 0
}
