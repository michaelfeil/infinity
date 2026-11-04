use crate::batch::PaddedInputs;
use crate::envelope::ResponseEnvelope;
use crate::executor::Executor;
use infinity_v3_backend_core::BackendError;
use pyo3::prelude::*;
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// One persistent Python worker per backend replica. Pending submissions are
/// bounded by the backend admission permits; the worker takes the GIL only
/// around interpreter calls and waits on the Rust queue without holding it.
pub(crate) struct Worker {
    state: Mutex<Option<State>>,
}

struct State {
    sender: Sender<Command>,
    handle: JoinHandle<()>,
}

struct SubmitJob {
    route: &'static str,
    payload: Arc<PaddedInputs>,
    envelope: ResponseEnvelope,
}

enum Command {
    Submit(SubmitJob),
    Health(SyncSender<Result<(), BackendError>>),
}

impl Worker {
    pub(crate) fn start(executor: Executor) -> Result<Self, BackendError> {
        let (sender, receiver) = channel();
        let handle = std::thread::Builder::new()
            .name("infinity-python".into())
            .spawn(move || worker_loop(executor, receiver))
            .map_err(|error| {
                BackendError::Start(format!("Could not start Python backend worker: {error}"))
            })?;
        Ok(Self {
            state: Mutex::new(Some(State { sender, handle })),
        })
    }

    pub(crate) fn submit(
        &self,
        route: &'static str,
        payload: Arc<PaddedInputs>,
        envelope: ResponseEnvelope,
    ) -> Result<(), BackendError> {
        self.sender()?
            .send(Command::Submit(SubmitJob {
                route,
                payload,
                envelope,
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
        let Some(state) = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take()
        else {
            return;
        };
        let State { sender, handle } = state;
        drop(sender);
        // Joining while holding the GIL would deadlock if the worker is waiting
        // to acquire it, so detach even when this thread already owns the GIL.
        let _ = Python::attach(|py| py.detach(|| handle.join()));
    }

    fn sender(&self) -> Result<Sender<Command>, BackendError> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .as_ref()
            .map(|state| state.sender.clone())
            .ok_or(BackendError::Unhealthy)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.close();
    }
}

fn worker_loop(executor: Executor, receiver: Receiver<Command>) {
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
        let envelope = match Py::new(py, job.envelope) {
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
