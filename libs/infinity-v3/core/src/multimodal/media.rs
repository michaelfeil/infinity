//! Bounded media retrieval. URLs and signed query strings never appear in errors.
use crate::TextEmbeddingsError;
use base64::{engine::general_purpose::STANDARD, Engine};
use reqwest::{redirect::Policy, Client, Url};
use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::Duration,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

type Result<T> = std::result::Result<T, TextEmbeddingsError>;
fn allocate_bytes(capacity: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| TextEmbeddingsError::Overloaded(tokio::sync::TryAcquireError::NoPermits))?;
    Ok(bytes)
}

fn invalid(message: &'static str) -> TextEmbeddingsError {
    TextEmbeddingsError::Tokenizer(message.into())
}

/// Reservations cover encoded bytes and processor allocations, then travel with GPU inputs.
#[derive(Clone, Debug)]
pub struct MediaBudget {
    semaphore: Arc<Semaphore>,
    units: u32,
}

impl MediaBudget {
    pub fn new(bytes: usize) -> Result<Self> {
        let units =
            u32::try_from(bytes / 1024).map_err(|_| invalid("Image memory budget is too large"))?;
        if units == 0 {
            return Err(invalid("Image memory budget must be at least 1 KiB"));
        }
        Ok(Self {
            semaphore: Arc::new(Semaphore::new(units as usize)),
            units,
        })
    }

    pub(crate) fn reserve(&self, bytes: usize) -> Result<OwnedSemaphorePermit> {
        let units = u32::try_from(bytes.div_ceil(1024))
            .map_err(|_| invalid("Image allocation is too large"))?;
        if units > self.units {
            return Err(TextEmbeddingsError::Validation(
                "Image allocation exceeds the configured memory budget".into(),
            ));
        }
        Ok(self.semaphore.clone().try_acquire_many_owned(units)?)
    }
}

pub(crate) struct ResolvedImage {
    pub bytes: Vec<u8>,
    // Includes the decoder's possible copy of the compressed input.
    _reservation: OwnedSemaphorePermit,
}

/// Empty allowlist permits inline data URLs only. Entries are exact DNS hostnames.
#[derive(Clone, Debug)]
pub struct MediaResolver {
    allowed_hosts: Arc<Vec<String>>,
    budget: MediaBudget,
    downloads: Arc<Semaphore>,
    max_bytes: usize,
    timeout: Duration,
}

impl MediaResolver {
    pub fn new(
        allowed_hosts: Vec<String>,
        budget: MediaBudget,
        max_bytes: usize,
        concurrency: usize,
        timeout: Duration,
    ) -> Result<Self> {
        if max_bytes == 0 || max_bytes > 64 * 1024 * 1024 || concurrency == 0 || timeout.is_zero() {
            return Err(invalid("Invalid image retrieval limits"));
        }
        let mut hosts = Vec::with_capacity(allowed_hosts.len());
        for host in allowed_hosts {
            let url = Url::parse(&format!("https://{host}/"))
                .map_err(|_| invalid("Invalid allowed image hostname"))?;
            if url.domain().is_none()
                || url.host_str() != Some(host.as_str())
                || !url.username().is_empty()
                || url.password().is_some()
                || url.port().is_some()
                || url.path() != "/"
                || url.query().is_some()
                || url.fragment().is_some()
            {
                return Err(invalid("Allowed image hosts must be exact lowercase DNS hostnames without ports or paths"));
            }
            hosts.push(host);
        }
        Ok(Self {
            allowed_hosts: Arc::new(hosts),
            budget,
            downloads: Arc::new(Semaphore::new(concurrency)),
            max_bytes,
            timeout,
        })
    }

    fn remote_url(&self, source: &str) -> Result<Url> {
        let url = Url::parse(source).map_err(|_| invalid("Invalid image URL"))?;
        if url.scheme() != "https"
            || !url.username().is_empty()
            || url.password().is_some()
            || url.port_or_known_default() != Some(443)
            || url.fragment().is_some()
            || !url
                .domain()
                .is_some_and(|host| self.allowed_hosts.iter().any(|allowed| allowed == host))
        {
            return Err(invalid(
                "Remote images require an explicitly allowed HTTPS hostname on port 443",
            ));
        }
        Ok(url)
    }

    pub(crate) async fn resolve(&self, source: String) -> Result<ResolvedImage> {
        tokio::time::timeout(self.timeout, self.resolve_inner(source))
            .await
            .map_err(|_| invalid("Image retrieval timed out"))?
    }

    async fn resolve_inner(&self, source: String) -> Result<ResolvedImage> {
        let slot = self
            .downloads
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| invalid("Image retrieval workers closed"))?;
        if source.starts_with("data:") {
            let limit = self.max_bytes;
            // Reserve the actual base64 capacity and the decoder's possible
            // compressed-input copy, rather than the configured maximum.
            let (_, capacity) = inline_data(&source, limit)?;
            let reservation = self.budget.reserve(capacity * 2)?;
            return tokio::task::spawn_blocking(move || {
                let _slot = slot;
                let bytes = decode_data_url(&source, limit)?;
                Ok(ResolvedImage {
                    bytes,
                    _reservation: reservation,
                })
            })
            .await
            .map_err(|_| invalid("Image data decoding failed"))?;
        }
        // Remote responses can omit Content-Length, so bound their full buffer
        // and the decoder's possible compressed-input copy before fetching.
        let reservation = self.budget.reserve(self.max_bytes * 2)?;
        let url = self.remote_url(&source)?;
        let host = url.domain().unwrap();
        // Pin the validated DNS results to this client's connection. Never re-resolve after validation.
        let addresses: Vec<SocketAddr> =
            tokio::time::timeout(self.timeout, tokio::net::lookup_host((host, 443)))
                .await
                .map_err(|_| invalid("Image DNS lookup timed out"))?
                .map_err(|_| invalid("Image DNS lookup failed"))?
                .collect();
        if addresses.is_empty()
            || addresses
                .iter()
                .any(|address| !public_address(address.ip()))
        {
            return Err(invalid("Image hostname resolved to a non-public address"));
        }
        let client = Client::builder()
            .https_only(true)
            .no_proxy()
            .redirect(Policy::none())
            .resolve_to_addrs(host, &addresses)
            .timeout(self.timeout)
            .connect_timeout(self.timeout)
            .build()
            .map_err(|_| invalid("Image HTTP client initialization failed"))?;
        // URL parsing preserves the signed query's ordering and escaping; never rebuild it.
        let mut response = client
            .get(url)
            .send()
            .await
            .map_err(|_| invalid("Image download failed or timed out"))?;
        if !response.status().is_success() {
            return Err(invalid(
                "Image download returned an unsuccessful HTTP status",
            ));
        }
        if response
            .content_length()
            .is_some_and(|length| length > self.max_bytes as u64)
        {
            return Err(TextEmbeddingsError::Validation(
                "Image exceeds the encoded byte limit".into(),
            ));
        }
        let mut bytes = allocate_bytes(self.max_bytes)?;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| invalid("Image download failed or timed out"))?
        {
            if chunk.len() > self.max_bytes - bytes.len() {
                return Err(TextEmbeddingsError::Validation(
                    "Image exceeds the encoded byte limit".into(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(ResolvedImage {
            bytes,
            _reservation: reservation,
        })
    }
}

fn inline_data(source: &str, max_bytes: usize) -> Result<(&str, usize)> {
    if source.len() > max_bytes.div_ceil(3) * 4 + 64 {
        return Err(invalid("Image exceeds the encoded byte limit"));
    }
    let (header, data) = source
        .split_once(',')
        .ok_or_else(|| invalid("Expected a base64 image data URL"))?;
    if !matches!(
        header,
        "data:image/png;base64"
            | "data:image/jpeg;base64"
            | "data:image/webp;base64"
            | "data:video/mp4;base64"
            | "data:video/webm;base64"
            | "data:video/quicktime;base64"
            | "data:video/x-matroska;base64"
            | "data:audio/wav;base64"
            | "data:audio/mpeg;base64"
    ) {
        return Err(invalid(
            "Inline images must be base64 PNG, JPEG, or WebP data URLs",
        ));
    }
    // Padding removes at most two bytes from the decoder's capacity estimate.
    // Reject oversized input before allocating, including invalid base64.
    let capacity = base64::decoded_len_estimate(data.len());
    if capacity.saturating_sub(2) > max_bytes {
        return Err(invalid("Image exceeds the encoded byte limit"));
    }
    Ok((data, capacity))
}

fn decode_data_url(source: &str, max_bytes: usize) -> Result<Vec<u8>> {
    let (data, capacity) = inline_data(source, max_bytes)?;
    let mut bytes = allocate_bytes(capacity)?;
    STANDARD
        .decode_vec(data, &mut bytes)
        .map_err(|_| invalid("Invalid image base64"))?;
    if bytes.len() > max_bytes {
        return Err(invalid("Image exceeds the encoded byte limit"));
    }
    Ok(bytes)
}

/// Conservative global-unicast policy: reject private, special-use and transition ranges.
fn public_address(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(ip) => {
            let [a, b, c, _] = ip.octets();
            !(a == 0
                || a == 10
                || a == 127
                || a >= 224
                || (a == 100 && (64..=127).contains(&b))
                || (a == 169 && b == 254)
                || (a == 172 && (16..=31).contains(&b))
                || (a == 192
                    && (b == 168 || (b == 0 && matches!(c, 0 | 2)) || (b == 88 && c == 99)))
                || (a == 198 && (matches!(b, 18 | 19) || (b == 51 && c == 100)))
                || (a == 203 && b == 0 && c == 113))
        }
        IpAddr::V6(ip) => {
            let segments = ip.segments();
            (segments[0] & 0xe000) == 0x2000
                && !(segments[0] == 0x2001 && (segments[1] < 0x0200 || segments[1] == 0x0db8))
                && segments[0] != 0x2002
                && segments[0] != 0x3fff
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn resolver() -> MediaResolver {
        MediaResolver::new(
            vec!["bucket.s3.us-east-1.amazonaws.com".into()],
            MediaBudget::new(4096).unwrap(),
            128,
            2,
            Duration::from_secs(1),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn occupied_retrieval_slots_queue_and_release_on_completion() {
        let resolver = resolver();
        let held = resolver
            .downloads
            .clone()
            .acquire_many_owned(2)
            .await
            .unwrap();
        let request = resolver.resolve("data:image/png;base64,AQID".into());
        tokio::pin!(request);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut request)
                .await
                .is_err()
        );
        drop(held);
        let image = request.await.unwrap();
        assert_eq!(image.bytes, [1, 2, 3]);
        assert_eq!(resolver.downloads.available_permits(), 2);
    }

    #[tokio::test]
    async fn queued_retrieval_respects_timeout_and_does_not_leak_permits() {
        let mut resolver = resolver();
        resolver.timeout = Duration::from_millis(10);
        let held = resolver
            .downloads
            .clone()
            .acquire_many_owned(2)
            .await
            .unwrap();
        let error = resolver
            .resolve("data:image/png;base64,AQID".into())
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("Image retrieval timed out"));
        assert_eq!(resolver.budget.semaphore.available_permits(), 4);
        drop(held);
        assert_eq!(resolver.downloads.available_permits(), 2);
        resolver.timeout = Duration::from_secs(1);
        assert!(resolver
            .resolve("data:image/png;base64,AQID".into())
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn concurrent_inline_images_reserve_actual_capacity() {
        let budget = MediaBudget::new(40 * 1024 * 1024).unwrap();
        let resolver = MediaResolver::new(
            vec![],
            budget.clone(),
            20 * 1024 * 1024,
            2,
            Duration::from_secs(5),
        )
        .unwrap();
        let mut requests = Vec::new();
        for _ in 0..32 {
            let resolver = resolver.clone();
            requests.push(tokio::spawn(async move {
                resolver.resolve("data:image/png;base64,AQID".into()).await
            }));
        }
        // Keep all compressed inputs alive, as preprocessing/queued inference do.
        let mut images = Vec::new();
        for request in requests {
            images.push(request.await.unwrap().unwrap());
        }
        assert!(images.iter().all(|image| image.bytes == [1, 2, 3]));
        assert_eq!(
            budget.semaphore.available_permits(),
            budget.units as usize - 32
        );
        drop(images);
        assert_eq!(budget.semaphore.available_permits(), budget.units as usize);
        assert_eq!(resolver.downloads.available_permits(), 2);
    }
    #[test]
    fn url_policy_preserves_signatures_and_rejects_bypasses() {
        let resolver = resolver();
        let signed = "https://bucket.s3.us-east-1.amazonaws.com/folder/image.png?X-Amz-Signature=a%2Fb%2B&x=2&x=1";
        assert_eq!(resolver.remote_url(signed).unwrap().as_str(), signed);
        for source in [
            "http://bucket.s3.us-east-1.amazonaws.com/image",
            "https://bucket.s3.us-east-1.amazonaws.com:444/image",
            "https://bucket.s3.us-east-1.amazonaws.com.evil.example/image",
            "https://user:secret@bucket.s3.us-east-1.amazonaws.com/image",
            "https://127.0.0.1/image",
            "file:///etc/passwd",
            "https://bucket.s3.us-east-1.amazonaws.com/image#secret",
        ] {
            assert!(resolver.remote_url(source).is_err());
        }
    }
    #[test]
    fn rejects_private_special_and_ipv4_embedded_addresses() {
        for address in [
            "127.0.0.1",
            "10.0.0.1",
            "169.254.169.254",
            "100.100.100.200",
            "192.168.1.1",
            "198.18.0.1",
            "224.1.1.1",
            "::1",
            "fc00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "64:ff9b::7f00:1",
            "2002:7f00:1::",
            "2001:db8::1",
        ] {
            assert!(!public_address(address.parse().unwrap()), "{address}");
        }
        for address in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
            assert!(public_address(address.parse().unwrap()));
        }
    }
    #[test]
    fn inline_limits_and_reservations_are_enforced() {
        assert_eq!(
            decode_data_url("data:image/png;base64,AQID", 3).unwrap(),
            [1, 2, 3]
        );
        assert!(decode_data_url("data:image/png;base64,AQID", 2).is_err());
        assert!(decode_data_url("data:text/plain;base64,AQID", 3).is_err());
        for (encoded, expected) in [("AQ==", vec![1]), ("AQI=", vec![1, 2])] {
            let url = format!("data:image/png;base64,{encoded}");
            assert_eq!(decode_data_url(&url, expected.len()).unwrap(), expected);
            assert!(decode_data_url(&url, expected.len() - 1).is_err());
        }
        assert!(matches!(
            allocate_bytes(usize::MAX),
            Err(TextEmbeddingsError::Overloaded(_))
        ));
        assert_eq!(allocate_bytes(3).unwrap().len(), 0);
        let budget = MediaBudget::new(2048).unwrap();
        let reservation = budget.reserve(1025).unwrap();
        assert!(matches!(
            budget.reserve(1),
            Err(TextEmbeddingsError::Overloaded(_))
        ));
        drop(reservation);
        assert!(budget.reserve(2048).is_ok());
    }
}
