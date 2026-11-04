use crate::batch::PaddedInputs;
use crate::envelope::{ResponseEnvelope, ResponseSink};
use crate::executor::{Executor, ExecutorConfig};
use infinity_v3_backend_core::BackendError;
use pyo3::prelude::*;
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// One persistent Python worker per backend replica. Submission only enqueues
/// into the Rust command channel, whose pending requests are bounded by the
/// backend admission permits. The worker acquires the GIL around interpreter
/// calls and waits for commands without holding it.
pub(crate) struct Worker {
    sender: Mutex<Option<Sender<Command>>>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

struct SubmitJob {
    route: &'static str,
    payload: Arc<PaddedInputs>,
    sink: ResponseSink,
}

enum Command {
    Submit(SubmitJob),
    Health(SyncSender<Result<(), BackendError>>),
}

impl Worker {
    pub(crate) fn start(config: ExecutorConfig) -> Result<Self, BackendError> {
        // Request admission already bounds pending work. A lazily allocated
        // channel avoids preallocating a user-configurable queue capacity.
        let (sender, receiver) = channel();
        let (started_sender, started_receiver) = sync_channel(1);
        let handle = std::thread::Builder::new()
            .name("infinity-python".into())
            .spawn(move || worker_loop(config, receiver, started_sender))
            .map_err(|error| {
                BackendError::Start(format!("Could not start Python backend worker: {error}"))
            })?;
        let started = Python::attach(|py| py.detach(move || started_receiver.recv()));
        match started {
            Ok(Ok(())) => Ok(Self {
                sender: Mutex::new(Some(sender)),
                handle: Mutex::new(Some(handle)),
            }),
            Ok(Err(error)) => {
                let _ = Python::attach(|py| py.detach(move || handle.join()));
                Err(error)
            }
            Err(_) => {
                let _ = Python::attach(|py| py.detach(move || handle.join()));
                Err(BackendError::Start(
                    "Python backend worker exited during startup".into(),
                ))
            }
        }
    }

    pub(crate) fn submit(
        &self,
        route: &'static str,
        payload: Arc<PaddedInputs>,
        sink: ResponseSink,
    ) -> Result<(), BackendError> {
        let sender = self.sender()?;
        sender
            .send(Command::Submit(SubmitJob {
                route,
                payload,
                sink,
            }))
            .map_err(|_| BackendError::Unhealthy)
    }

    pub(crate) fn health(&self) -> Result<(), BackendError> {
        let sender = self.sender()?;
        let (response, receiver) = sync_channel(1);
        Python::attach(|py| py.detach(move || sender.send(Command::Health(response))))
            .map_err(|_| BackendError::Unhealthy)?;
        Python::attach(|py| py.detach(move || receiver.recv()))
            .map_err(|_| BackendError::Unhealthy)?
    }

    pub(crate) fn close(&self) {
        {
            let mut sender = self
                .sender
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            sender.take();
        }
        let handle = {
            let mut handle = self
                .handle
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            handle.take()
        };
        if let Some(handle) = handle {
            // Joining while the GIL is held would deadlock if the worker is
            // waiting to acquire it. Detach for the join even when this thread
            // already owns the GIL.
            let _ = Python::attach(|py| py.detach(|| handle.join()));
        }
    }

    fn sender(&self) -> Result<Sender<Command>, BackendError> {
        let sender = self
            .sender
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        sender.clone().ok_or(BackendError::Unhealthy)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.close();
    }
}

fn worker_loop(
    config: ExecutorConfig,
    receiver: Receiver<Command>,
    started: SyncSender<Result<(), BackendError>>,
) {
    let executor = match Executor::new(config) {
        Ok(executor) => {
            let _ = started.send(Ok(()));
            executor
        }
        Err(error) => {
            let _ = started.send(Err(error));
            return;
        }
    };
    // Blocking recv waits on the Rust queue without holding the GIL. Dropping
    // the last sender drains commands already accepted before closing.
    while let Ok(command) = receiver.recv() {
        match command {
            Command::Submit(job) => execute(&executor, job),
            Command::Health(response) => {
                let _ = response.send(executor.health());
            }
        }
    }
    Python::attach(|py| {
        executor.close_with_py(py);
        drop(executor);
    });
}

fn execute(executor: &Executor, job: SubmitJob) {
    Python::attach(|py| {
        let envelope = match ResponseEnvelope::from_sink(py, job.sink) {
            Ok(envelope) => envelope,
            Err(error) => {
                tracing::error!(%error, "Could not create Python response envelope");
                return;
            }
        };
        let payload = match PaddedInputs::to_python(job.payload, py) {
            Ok(payload) => payload,
            Err(error) => {
                let _ = envelope.borrow_mut(py).fail(error);
                return;
            }
        };
        if let Err(error) = executor.submit(py, job.route, envelope.clone_ref(py), payload) {
            let _ = envelope.borrow_mut(py).fail(error);
        }
    });
}
