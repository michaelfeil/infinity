// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Stream capture for fixed-address oxide workspaces. Captured allocations and
//! kernel modules must outlive graph launches; the engine drains its stream
//! before retiring a workspace or executable.

use cuda_core::{sys, Context, IntoResult, Stream};
use infinity_v3_backend_core::BackendError;
use std::{marker::PhantomData, ptr, rc::Rc, sync::Arc};

fn error(operation: &str, error: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(format!("CUDA graph {operation}: {error}"))
}

/// A thread-local capture cannot move to another host thread before ending.
/// Dropping it after an enqueue failure still ends the capture and releases any
/// graph the driver produced, so a failed request does not strand the stream.
pub(crate) struct Capture {
    stream: Arc<Stream>,
    active: bool,
    graph: sys::CUgraph,
    _thread_local: PhantomData<Rc<()>>,
}

impl Capture {
    pub(crate) fn begin(stream: Arc<Stream>) -> Result<Self, BackendError> {
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| error("bind for capture", e))?;
        // SAFETY: the stream/context are retained. The !Send capture guard
        // ensures begin/end run on the same thread required by this mode.
        unsafe {
            sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        }
        .result()
        .map_err(|e| error("begin capture", e))?;
        Ok(Self {
            stream,
            active: true,
            graph: ptr::null_mut(),
            _thread_local: PhantomData,
        })
    }

    pub(crate) fn finish(mut self) -> Result<GraphExec, BackendError> {
        let context = self.stream.context().clone();
        context
            .bind_to_thread()
            .map_err(|e| error("bind to finish capture", e))?;
        // SAFETY: this guard owns the active capture on the current thread.
        // EndCapture terminates capture even when it reports invalidation.
        let ended = unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut self.graph) };
        self.active = false;
        ended.result().map_err(|e| error("end capture", e))?;
        if self.graph.is_null() {
            return Err(error("end capture", "driver returned a null graph"));
        }
        let mut raw = ptr::null_mut();
        // SAFETY: self owns a successfully captured graph in this context.
        unsafe { sys::cuGraphInstantiateWithFlags(&mut raw, self.graph, 0) }
            .result()
            .map_err(|e| error("instantiate", e))?;
        if raw.is_null() {
            return Err(error("instantiate", "driver returned a null executable"));
        }
        let executable = GraphExec { raw, context };
        // The executable owns its instantiated nodes independently. Clear our
        // ownership before destroy so a reported asynchronous error cannot
        // cause a second destruction attempt in Drop.
        let graph = std::mem::replace(&mut self.graph, ptr::null_mut());
        // SAFETY: this graph is no longer needed after successful instantiation.
        unsafe { sys::cuGraphDestroy(graph) }
            .result()
            .map_err(|e| error("destroy captured graph", e))?;
        Ok(executable)
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let context = self.stream.context();
        if let Err(e) = context.bind_to_thread() {
            context.record_err(Err::<(), _>(e));
            return;
        }
        if self.active {
            // SAFETY: a live !Send guard owns capture on this host thread.
            let ended =
                unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut self.graph) };
            self.active = false;
            context.record_err(ended.result());
        }
        if !self.graph.is_null() {
            let graph = std::mem::replace(&mut self.graph, ptr::null_mut());
            // SAFETY: graph ownership remains here if finish did not consume it.
            context.record_err(unsafe { sys::cuGraphDestroy(graph) }.result());
        }
    }
}

pub(crate) struct GraphExec {
    raw: sys::CUgraphExec,
    context: Arc<Context>,
}
// SAFETY: CUDA graph executable handles are not host-thread-affine. The engine
// serializes launches and destruction behind its Mutex and retains all captured
// allocations/modules until its stream is drained. The primary context is kept
// alive here and rebound on every driver operation. No Sync implementation is
// provided for concurrent access outside that engine discipline.
unsafe impl Send for GraphExec {}

impl GraphExec {
    pub(crate) fn launch(&self, stream: &Stream) -> Result<(), BackendError> {
        if self.context.cu_ctx() != stream.context().cu_ctx() {
            return Err(error("launch", "stream belongs to another CUDA context"));
        }
        self.context
            .bind_to_thread()
            .map_err(|e| error("bind for launch", e))?;
        // SAFETY: this live executable and stream share a retained context;
        // the engine retains the fixed workspace and serializes execution.
        unsafe { sys::cuGraphLaunch(self.raw, stream.cu_stream()) }
            .result()
            .map_err(|e| error("launch", e))
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if let Err(e) = self.context.bind_to_thread() {
            self.context.record_err(Err::<(), _>(e));
            return;
        }
        // SAFETY: the owning engine/workspace drains launches before teardown.
        self.context
            .record_err(unsafe { sys::cuGraphExecDestroy(self.raw) }.result());
    }
}
