//! Native transports for the Multiuser Xtra's game socket.
//!
//! On the browser (`wasm32`) the Xtra is backed by `web_sys::WebSocket` (see the
//! parent module). Native targets have no browser socket, so this module
//! provides a small [`NetSocket`] abstraction with two backends the headless
//! harness selects between:
//!
//! - [`MockNetSocket`] — captures sent bytes and is fed inbound bytes in-process
//!   (the existing conformance-harness path).
//! - [`TcpNetSocket`] — a real `async-std` TCP connection, for end-to-end tests
//!   against a live server over loopback.
//!
//! Following the engine's multi-backend convention (cf. `DynamicRenderer` in
//! `rendering_gpu`), the two are unified by the [`NetSocketBackend`] enum rather
//! than a `Box<dyn>`.
//!
//! The operations are synchronous and queue-oriented to match the VM's
//! single-threaded, manually-pumped model: `connect`/`send` only record intent,
//! and the harness performs the real I/O out-of-band (the mock via injection,
//! TCP via [`TcpNetSocket::drive`], which it `.await`s — so no background task or
//! nested executor is needed). (bobba habbo-oracle)

use std::collections::VecDeque;
use std::time::Duration;

use async_std::io::{ReadExt, WriteExt};
use async_std::net::TcpStream;

use crate::player::ScriptError;

/// Upper bound on how long [`TcpNetSocket::drive`] waits for a server reply
/// before giving up. Generous on purpose: during the handshake the server
/// answers each client send within microseconds on loopback, so this never
/// trips in normal flow — it exists only so a stalled or misbehaving server
/// fails the caller promptly instead of hanging indefinitely.
const READ_BACKSTOP: Duration = Duration::from_secs(5);

/// A transport backing the Multiuser Xtra's game socket on native targets.
pub trait NetSocket {
    /// Record the server to connect to. Native connects are lazy — no I/O here.
    fn connect(&mut self, host: &str, port: i32) -> Result<(), ScriptError>;
    /// Enqueue an already-framed client→server message for delivery.
    fn send(&mut self, bytes: Vec<u8>) -> Result<(), ScriptError>;
    /// Tear the connection down.
    fn disconnect(&mut self);
    /// Whether the transport considers itself connected.
    fn is_connected(&self) -> bool;
}

/// In-process transport for the conformance harness: sent bytes are captured for
/// the harness to inspect, and inbound bytes are injected straight into the
/// Xtra's message queue (see `MultiuserXtraManager::mock_inject`).
#[derive(Default)]
pub struct MockNetSocket {
    captured_out: VecDeque<Vec<u8>>,
    connected: bool,
}

impl MockNetSocket {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drain the bytes the client has sent (captured in place of a real socket).
    pub fn take_sent(&mut self) -> Vec<Vec<u8>> {
        self.captured_out.drain(..).collect()
    }
}

impl NetSocket for MockNetSocket {
    fn connect(&mut self, _host: &str, _port: i32) -> Result<(), ScriptError> {
        self.connected = true;
        Ok(())
    }

    fn send(&mut self, bytes: Vec<u8>) -> Result<(), ScriptError> {
        self.captured_out.push_back(bytes);
        Ok(())
    }

    fn disconnect(&mut self) {
        self.connected = false;
    }

    fn is_connected(&self) -> bool {
        self.connected
    }
}

/// Real TCP transport (async-std), for end-to-end conformance against a live
/// server. `connect`/`send` only stash intent; [`drive`](Self::drive) performs
/// the actual connect/write/read and is awaited by the harness, so the VM's
/// single-threaded model is preserved (no background task, no nested executor).
pub struct TcpNetSocket {
    host: String,
    port: i32,
    stream: Option<TcpStream>,
    outbound: VecDeque<Vec<u8>>,
}

impl Default for TcpNetSocket {
    fn default() -> Self {
        Self {
            host: String::new(),
            port: 0,
            stream: None,
            outbound: VecDeque::new(),
        }
    }
}

impl TcpNetSocket {
    pub fn new() -> Self {
        Self::default()
    }

    /// Move the live stream out (so the caller can `.await` I/O on it without
    /// holding a borrow of the global Xtra manager).
    pub fn take_stream(&mut self) -> Option<TcpStream> {
        self.stream.take()
    }

    /// Return the stream after I/O.
    pub fn restore_stream(&mut self, stream: Option<TcpStream>) {
        self.stream = stream;
    }

    /// Drain the queued outbound messages.
    pub fn take_outbound(&mut self) -> Vec<Vec<u8>> {
        self.outbound.drain(..).collect()
    }

    /// The configured `(host, port)`.
    pub fn target(&self) -> (String, i32) {
        (self.host.clone(), self.port)
    }

    /// Connect if needed, write every queued message, then **block** until the
    /// server responds and return that batch of reply bytes.
    ///
    /// The game protocol is strictly call-and-response during the handshake:
    /// every read follows a client send the server is obliged to answer, so
    /// waiting on the socket (rather than polling on a short timeout and making
    /// the harness busy-spin) is the natural event-driven model. A quiescent
    /// socket therefore means one of two things, both terminal: the server
    /// closed after the final reply (surfaced as [`UnexpectedEof`]), or
    /// something stalled — for which [`READ_BACKSTOP`] turns an otherwise
    /// indefinite hang into a prompt error the caller can fail on.
    ///
    /// Operates on an owned `stream` + `outbound` so the caller holds no manager
    /// borrow across an `.await`.
    ///
    /// [`UnexpectedEof`]: std::io::ErrorKind::UnexpectedEof
    pub async fn drive(
        stream: &mut Option<TcpStream>,
        host: &str,
        port: i32,
        outbound: Vec<Vec<u8>>,
    ) -> std::io::Result<Vec<u8>> {
        if stream.is_none() {
            let socket = TcpStream::connect((host, port as u16)).await?;
            // Send small handshake frames immediately — Nagle's algorithm would
            // otherwise stall tiny request/reply packets against delayed ACKs.
            socket.set_nodelay(true)?;
            *stream = Some(socket);
        }
        let socket = stream.as_mut().expect("stream connected just above");

        for message in outbound {
            socket.write_all(&message).await?;
        }
        socket.flush().await?;

        let mut buf = [0u8; 8192];
        match async_std::future::timeout(READ_BACKSTOP, socket.read(&mut buf)).await {
            Ok(Ok(0)) => Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "server closed the connection",
            )),
            Ok(Ok(n)) => Ok(buf[..n].to_vec()),
            Ok(Err(e)) => Err(e),
            Err(_timed_out) => Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "no server response within the read backstop",
            )),
        }
    }
}

impl NetSocket for TcpNetSocket {
    fn connect(&mut self, host: &str, port: i32) -> Result<(), ScriptError> {
        self.host = host.to_string();
        self.port = port;
        Ok(())
    }

    fn send(&mut self, bytes: Vec<u8>) -> Result<(), ScriptError> {
        self.outbound.push_back(bytes);
        Ok(())
    }

    fn disconnect(&mut self) {
        self.stream = None;
    }

    fn is_connected(&self) -> bool {
        self.stream.is_some()
    }
}

/// The native transport backend, dispatched by enum (the engine's convention for
/// swappable implementations, cf. `DynamicRenderer`).
pub enum NetSocketBackend {
    Mock(MockNetSocket),
    Tcp(TcpNetSocket),
}

impl NetSocket for NetSocketBackend {
    fn connect(&mut self, host: &str, port: i32) -> Result<(), ScriptError> {
        match self {
            NetSocketBackend::Mock(socket) => socket.connect(host, port),
            NetSocketBackend::Tcp(socket) => socket.connect(host, port),
        }
    }

    fn send(&mut self, bytes: Vec<u8>) -> Result<(), ScriptError> {
        match self {
            NetSocketBackend::Mock(socket) => socket.send(bytes),
            NetSocketBackend::Tcp(socket) => socket.send(bytes),
        }
    }

    fn disconnect(&mut self) {
        match self {
            NetSocketBackend::Mock(socket) => socket.disconnect(),
            NetSocketBackend::Tcp(socket) => socket.disconnect(),
        }
    }

    fn is_connected(&self) -> bool {
        match self {
            NetSocketBackend::Mock(socket) => socket.is_connected(),
            NetSocketBackend::Tcp(socket) => socket.is_connected(),
        }
    }
}
