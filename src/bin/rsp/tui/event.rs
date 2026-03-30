use std::{sync::mpsc, time::Duration};

use crossterm::event::{self, Event as CrosstermEvent, KeyEvent, MouseEvent};

/// Application events combining terminal input and simulation messages.
#[derive(Debug)]
pub enum Event<T> {
    /// A key press event
    Key(KeyEvent),
    /// A mouse event (click, scroll, etc.)
    Mouse(MouseEvent),
    /// A periodic tick for UI refresh
    Tick,
    /// A simulation message from the background thread
    Sim(SimMessage<T>),
    /// Terminal resize
    Resize,
}

/// Error from a simulation background thread.
#[derive(Debug, thiserror::Error)]
pub enum SimError {
    /// The simulation panicked unexpectedly.
    #[error("simulation panicked")]
    Panic,
    /// The comparison returned a domain error.
    #[error("comparison failed: {source}")]
    ComparisonFailed {
        #[source]
        source: rs_poker::arena::comparison::ComparisonError,
    },
    /// Too many consecutive game setup failures.
    #[error("too many consecutive failures ({consecutive_failures})")]
    TooManyFailures { consecutive_failures: usize },
}

/// Messages sent from the simulation background thread.
#[derive(Debug)]
pub enum SimMessage<T> {
    /// A game result to incorporate into state
    GameResult(T),
    /// Simulation has completed
    Completed,
    /// An error occurred during simulation
    Error(SimError),
}

/// Event handler that merges crossterm events with simulation channel messages.
pub struct EventHandler<T> {
    rx: mpsc::Receiver<SimMessage<T>>,
    tick_rate: Duration,
}

impl<T> EventHandler<T> {
    /// Create a new event handler with the given simulation receiver and tick rate.
    pub fn new(rx: mpsc::Receiver<SimMessage<T>>, tick_rate: Duration) -> Self {
        Self { rx, tick_rate }
    }

    /// Poll for the next event, blocking up to tick_rate duration.
    ///
    /// Returns `Tick` if no other event arrives within the tick interval.
    pub fn next(&self) -> std::io::Result<Event<T>> {
        // First, drain any pending sim messages (prioritize data updates)
        if let Ok(msg) = self.rx.try_recv() {
            return Ok(Event::Sim(msg));
        }

        // Poll for crossterm events with timeout
        if event::poll(self.tick_rate)? {
            match event::read()? {
                CrosstermEvent::Key(key) => return Ok(Event::Key(key)),
                CrosstermEvent::Mouse(mouse) => return Ok(Event::Mouse(mouse)),
                CrosstermEvent::Resize(_, _) => return Ok(Event::Resize),
                _ => {}
            }
        }

        Ok(Event::Tick)
    }

    /// Try to receive a single pending simulation message, if any.
    pub fn try_recv_sim(&self) -> Option<SimMessage<T>> {
        self.rx.try_recv().ok()
    }
}
