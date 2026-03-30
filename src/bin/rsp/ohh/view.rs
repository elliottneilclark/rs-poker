use std::sync::mpsc;

use clap::Args;

use crate::ohh::{reader, stats};
use crate::tui::{
    TuiFlags,
    app::{self, App},
    event::{EventHandler, SimMessage},
    hand_store::HandStore,
    state::GameResult,
};

/// View an Open Hand History file or directory
#[derive(Args, Debug)]
pub struct ViewArgs {
    /// Path to an .ohh file or a directory of .ohh files
    path: std::path::PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum ViewError {
    #[error("failed to read OHH file: {0}")]
    Reader(#[from] reader::ReaderError),
    #[error("TUI error: {0}")]
    Tui(#[from] std::io::Error),
    #[error("failed to build hand store: {0}")]
    HandStore(#[from] crate::tui::hand_store::HandStoreError),
}

pub fn run(args: ViewArgs, tui_flags: &TuiFlags) -> Result<(), ViewError> {
    let is_dir = args.path.is_dir();

    let hands = if is_dir {
        reader::read_ohh_dir(&args.path)?
    } else {
        reader::read_ohh_file(&args.path)?
    };

    if hands.is_empty() {
        println!("No hands found.");
        return Ok(());
    }

    if !tui_flags.should_use_tui() {
        print_text_summary(&hands);
        return Ok(());
    }

    let hand_store = if is_dir {
        HandStore::from_existing_dir(&args.path)?
    } else {
        HandStore::from_existing(&args.path)?
    };

    let mut state = stats::build_state_from_hands(&hands);
    state.live = false;
    let mut app = App::new_with_state(state, hand_store);

    // Create a dummy channel (no live updates for OHH viewer)
    let (_tx, rx) = mpsc::channel::<SimMessage<GameResult>>();
    let handler = EventHandler::new(rx, std::time::Duration::from_millis(33));

    app::run_app(&mut app, &handler)?;

    Ok(())
}

fn print_text_summary(hands: &[rs_poker::open_hand_history::HandHistory]) {
    let mut state = stats::build_state_from_hands(hands);
    let agents = state.agent_display_data();

    println!(
        "=== Hand History Summary ({} games) ===\n",
        state.games_completed
    );

    println!(
        "{:<20} {:>10} {:>8} {:>7} {:>7} {:>7} {:>7} {:>6}",
        "Agent", "Profit", "Games", "Win%", "ROI%", "VPIP%", "PFR%", "AF"
    );
    println!("{}", "-".repeat(82));

    for agent in &agents {
        let win_pct = if agent.games_played > 0 {
            agent.wins as f32 / agent.games_played as f32 * 100.0
        } else {
            0.0
        };
        println!(
            "{:<20} {:>+10.1} {:>8} {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.2}",
            agent.name,
            agent.total_profit,
            agent.games_played,
            win_pct,
            agent.roi_percent,
            agent.vpip_percent,
            agent.pfr_percent,
            agent.aggression_factor,
        );
    }

    println!(
        "\nStreet Distribution: Preflop={} Flop={} Turn={} River={} Showdown={}",
        state.street_dist.preflop,
        state.street_dist.flop,
        state.street_dist.turn,
        state.street_dist.river,
        state.street_dist.showdown,
    );
}
