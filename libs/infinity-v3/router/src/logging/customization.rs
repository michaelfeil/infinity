use std::fmt;

use tracing::{Event, Subscriber};
use tracing_subscriber::fmt::format::{FormatEvent, FormatFields, Writer};
use tracing_subscriber::fmt::time::{FormatTime, SystemTime};
use tracing_subscriber::fmt::{FmtContext, FormattedFields};
use tracing_subscriber::registry::LookupSpan;

pub(super) const LOG_PREFIX: Option<&str> = option_env!("INFINITY_LOG_PREFIX");

/// Customization affects rendered targets only; filtering and telemetry retain their metadata.
pub(super) struct LogFormat<F> {
    inner: F,
    json: bool,
}

impl<F> LogFormat<F> {
    pub(super) fn text(inner: F) -> Self {
        Self { inner, json: false }
    }

    pub(super) fn json(inner: F) -> Self {
        Self { inner, json: true }
    }
}

fn customized_target(target: &str, prefix: &str) -> Option<String> {
    let suffix = target.strip_prefix("infinity")?;
    // Match our crate namespace, never similarly named dependencies.
    if suffix.is_empty() || suffix.starts_with('_') || suffix.starts_with("::") {
        Some(format!("{prefix}{suffix}"))
    } else {
        None
    }
}

impl<S, N, F> FormatEvent<S, N> for LogFormat<F>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
    F: FormatEvent<S, N>,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let prefix = match LOG_PREFIX {
            None | Some("infinity") => return self.inner.format_event(ctx, writer, event),
            Some(prefix) => prefix,
        };
        let target = customized_target(event.metadata().target(), prefix);
        if self.json {
            let mut rendered = String::new();
            self.inner
                .format_event(ctx, Writer::new(&mut rendered), event)?;
            // The standard formatter emits the event once without its target.
            // Append only that metadata field, preserving every other JSON byte.
            let body = rendered.strip_suffix("}\n").ok_or(fmt::Error)?;
            let target = target.as_deref().unwrap_or(event.metadata().target());
            let encoded = serde_json::to_string(target).map_err(|_| fmt::Error)?;
            return writeln!(writer, "{body},\"target\":{encoded}}}");
        }
        let Some(target) = target else {
            return self.inner.format_event(ctx, writer, event);
        };

        // Render customized text directly so target-like strings in messages and span
        // fields are never rewritten. Default builds use the original formatter.
        SystemTime.format_time(&mut writer)?;
        write!(writer, " {:>5} ", event.metadata().level())?;
        if let Some(scope) = ctx.event_scope() {
            for span in scope.from_root() {
                write!(writer, "{}", span.name())?;
                let extensions = span.extensions();
                if let Some(fields) = extensions.get::<FormattedFields<N>>() {
                    if !fields.is_empty() {
                        write!(writer, "{{{fields}}}")?;
                    }
                }
                writer.write_char(':')?;
            }
            writer.write_char(' ')?;
        }
        write!(writer, "{target}: ")?;
        if let Some(file) = event.metadata().file() {
            write!(writer, "{file}:")?;
            if let Some(line) = event.metadata().line() {
                write!(writer, "{line}:")?;
            }
            writer.write_char(' ')?;
        }
        ctx.format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::{EnvFilter, Layer};

    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl io::Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Capture {
        fn output(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    fn emit() {
        let span = tracing::info_span!(target: "infinity_v3_core", "request", original = "infinity_v3_core::backend_pool");
        let _entered = span.enter();
        tracing::info!(target: "infinity_v3_core::backend_pool", count = 2, target = "user field", exact = u64::MAX,
            "keep infinity_v3_core::backend_pool unchanged in messages");
        tracing::info!(target: "dependency::pool", "dependency event");
        tracing::info!(target: "infinityother::pool", "similarly named dependency");
        tracing::info!(target: "unrelated", "filtered event");
    }

    #[test]
    fn rendered_targets_preserve_fields_dependencies_and_original_filters() {
        for json in [false, true] {
            let capture = Capture::default();
            let sink = capture.clone();
            let layer = tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_file(true)
                .with_line_number(true)
                .with_writer(move || sink.clone());
            let layer = if json {
                layer
                    .json()
                    .flatten_event(true)
                    .with_target(matches!(LOG_PREFIX, None | Some("infinity")))
                    .map_event_format(LogFormat::json)
                    .boxed()
            } else {
                layer.map_event_format(LogFormat::text).boxed()
            };
            let subscriber = tracing_subscriber::registry()
                .with(EnvFilter::new(
                    "off,infinity_v3_core=info,dependency=info,infinityother=info",
                ))
                .with(layer);
            tracing::subscriber::with_default(subscriber, emit);
            let output = capture.output();
            let expected = format!("{}_v3_core::backend_pool", LOG_PREFIX.unwrap_or("infinity"));
            if json {
                let events: Vec<serde_json::Value> = output
                    .lines()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
                assert_eq!(events.len(), 3);
                assert_eq!(events[0]["target"], expected);
                assert_eq!(events[0]["count"], 2);
                assert_eq!(events[0]["exact"].as_u64(), Some(u64::MAX));
                assert_eq!(
                    events[0]["span"]["original"],
                    "infinity_v3_core::backend_pool"
                );
                assert_eq!(
                    events[0]["message"],
                    "keep infinity_v3_core::backend_pool unchanged in messages"
                );
                assert_eq!(events[1]["target"], "dependency::pool");
                assert_eq!(events[2]["target"], "infinityother::pool");
                assert!(events[0]["line_number"].is_number());
            } else {
                assert!(output.contains(&format!("{expected}: ")));
                assert!(
                    output.contains("keep infinity_v3_core::backend_pool unchanged in messages")
                );
                assert!(output.contains("original=\"infinity_v3_core::backend_pool\""));
                assert!(output.contains("dependency::pool:"));
                assert!(output.contains("infinityother::pool:"));
            }
            assert!(!output.contains("filtered event"));
        }
    }
}
