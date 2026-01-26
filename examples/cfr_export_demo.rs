mod common;

use rs_poker::arena::GameState;
use rs_poker::arena::cfr::{
    CFRState, ExportFormat, NodeData, PlayerData, TerminalData, export_cfr_state,
};
use std::path::Path;

/// Creates an example CFR state tree for demonstration purposes.
/// The tree represents a simple poker game with two players, including fold,
/// call, and raise actions.
fn create_example_cfr() -> CFRState {
    // Create a game state with 2 players
    let game_state = GameState::new_starting(vec![100.0; 2], 10.0, 5.0, 0.0, 0);
    let mut cfr_state = CFRState::new(game_state);

    // Root -> Player 0 decision
    let player0_node = NodeData::Player(PlayerData {
        regret_matcher: None,
        player_idx: 0,
    });
    let player0_idx = cfr_state.add(0, 0, player0_node);

    // Player 0 fold
    let terminal_fold = NodeData::Terminal(TerminalData::new(-10.0));
    let _fold_idx = cfr_state.add(player0_idx, 0, terminal_fold);

    // Player 0 call
    let player0_call = cfr_state.add(player0_idx, 1, NodeData::Chance);

    // Player 0 raise
    let player0_raise = cfr_state.add(player0_idx, 2, NodeData::Chance);

    // After call - chance node (dealing flop)
    for i in 0..3 {
        // Create 3 sample card possibilities
        let player1_node = NodeData::Player(PlayerData {
            regret_matcher: None,
            player_idx: 1,
        });
        let player1_idx = cfr_state.add(player0_call, i, player1_node);

        // Player 1 fold
        let p1_fold_terminal = NodeData::Terminal(TerminalData::new(15.0));
        cfr_state.add(player1_idx, 0, p1_fold_terminal);

        // Player 1 call
        let p1_call_terminal = NodeData::Terminal(TerminalData::new(5.0));
        cfr_state.add(player1_idx, 1, p1_call_terminal);
    }

    // After raise - player 1 decision
    let player1_vs_raise = NodeData::Player(PlayerData {
        regret_matcher: None,
        player_idx: 1,
    });
    let player1_vs_raise_idx = cfr_state.add(player0_raise, 0, player1_vs_raise);

    // Player 1 fold vs raise
    let p1_fold_vs_raise = NodeData::Terminal(TerminalData::new(20.0));
    cfr_state.add(player1_vs_raise_idx, 0, p1_fold_vs_raise);

    // Player 1 call vs raise - goes to another chance node
    let chance_after_call_vs_raise = cfr_state.add(player1_vs_raise_idx, 1, NodeData::Chance);

    // Final terminal node after chance
    let final_terminal = NodeData::Terminal(TerminalData::new(30.0));
    cfr_state.add(chance_after_call_vs_raise, 0, final_terminal);

    // Increment some counts to simulate traversals
    cfr_state.increment_count(player0_idx, 1).unwrap(); // Call was taken once
    cfr_state.increment_count(player0_idx, 2).unwrap(); // Raise was taken twice
    cfr_state.increment_count(player0_idx, 2).unwrap();

    cfr_state
}

/// This example demonstrates the export functionality of the CFR state.
/// It creates a simple CFR tree and exports it to DOT, PNG, and SVG formats.
fn main() {
    common::init_tracing_from_env();

    // Create an example CFR state
    let cfr_state = create_example_cfr();

    // Export the CFR state to various formats
    let export_path = Path::new("examples/exports/cfr_tree");

    println!("Exporting CFR state to {}", export_path.display());

    // Export to DOT format
    if let Err(e) = export_cfr_state(&cfr_state, export_path, ExportFormat::Dot) {
        eprintln!("Error exporting to DOT: {e}");
    } else {
        println!(
            "Successfully exported to DOT format: {}.dot",
            export_path.display()
        );
    }

    // Export to PNG format (requires Graphviz)
    if let Err(e) = export_cfr_state(&cfr_state, export_path, ExportFormat::Png) {
        eprintln!("Error exporting to PNG: {e}");
    } else {
        println!(
            "Successfully exported to PNG format: {}.png",
            export_path.display()
        );
    }

    // Export to SVG format (requires Graphviz)
    if let Err(e) = export_cfr_state(&cfr_state, export_path, ExportFormat::Svg) {
        eprintln!("Error exporting to SVG: {e}");
    } else {
        println!(
            "Successfully exported to SVG format: {}.svg",
            export_path.display()
        );
    }

    // Export to all formats
    if let Err(e) = export_cfr_state(&cfr_state, export_path, ExportFormat::All) {
        eprintln!("Error exporting to all formats: {e}");
    } else {
        println!("Successfully exported to all formats");
    }

    println!("Files have been created in the examples/exports directory");
}
