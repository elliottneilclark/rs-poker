use std::io::{self, Stdout};

use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, prelude::CrosstermBackend};

pub type Tui = Terminal<CrosstermBackend<Stdout>>;

/// RAII guard that disables raw mode on drop.
///
/// Used during [`setup_terminal`] to unwind terminal state if a subsequent
/// initialization step fails. On the happy path the guard is forgotten via
/// [`std::mem::forget`], transferring cleanup responsibility to the caller
/// (who must later call [`restore_terminal`]).
struct RawModeGuard;

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

/// Initialize the terminal for TUI rendering.
///
/// Enables raw mode, enters alternate screen, enables mouse capture,
/// and installs a panic hook that restores the terminal before aborting.
/// If any step after [`enable_raw_mode`] fails, raw mode is automatically
/// disabled before propagating the error so the user's shell doesn't end
/// up scrambled.
pub fn setup_terminal() -> io::Result<Tui> {
    install_panic_hook();
    enable_raw_mode()?;
    // Any early return from here on drops the guard, which restores the
    // terminal by disabling raw mode.
    let guard = RawModeGuard;

    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

    let backend = CrosstermBackend::new(stdout);
    let terminal = match Terminal::new(backend) {
        Ok(t) => t,
        Err(e) => {
            // Roll back what we just enabled before the guard disables
            // raw mode.
            let _ = execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
            return Err(e);
        }
    };

    // Success: caller now owns the terminal and is responsible for
    // calling [`restore_terminal`]. Skip the guard's Drop impl.
    std::mem::forget(guard);
    Ok(terminal)
}

/// Restore the terminal to its original state.
pub fn restore_terminal(terminal: &mut Tui) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        DisableMouseCapture,
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;
    Ok(())
}

/// Install a panic hook that restores the terminal before the default hook runs.
fn install_panic_hook() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
        original_hook(panic_info);
    }));
}
