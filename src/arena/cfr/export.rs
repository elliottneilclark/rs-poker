//! CFR State Tree Visualization
//! 
//! This module provides functionality to visualize the Counterfactual Regret Minimization (CFR)
//! state tree in various formats. The visualization helps in understanding and debugging the
//! poker game tree structure and decision points.
//!
//! # Features
//!
//! - Multiple export formats:
//!   - DOT (Graphviz format)
//!   - PNG (static image)
//!   - SVG (scalable, interactive in browsers)
//!
//! - Node visualization:
//!   - Root nodes: Light blue, double octagon shape
//!   - Chance nodes: Light green, ellipse shape - shows possible card deals
//!   - Player nodes: Coral color, box shape - shows player seat and action choices
//!   - Terminal nodes: Light grey, hexagon shape - shows utility values
//!
//! - Edge information:
//!   - For chance nodes: Labels show which cards could be dealt
//!   - For player nodes: Labels show actions (fold at index 0)
//!   - Edge thickness indicates frequency of use
//!
//! # Usage
//!
//! ```rust
//! use rs_poker::arena::cfr::{CFRState, export_cfr_state};
//! use std::path::Path;
//!
//! // Create your CFR state...
//! let state = CFRState::new(/* ... */);
//!
//! // Export to all formats
//! export_cfr_state(&state, Path::new("game_tree"), "all").unwrap();
//! ```

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use crate::core::Card;

use super::{CFRState, NodeData};

/// Exports the CFR state tree to a DOT (Graphviz) file.
/// 
/// This function creates a DOT file representation of the CFR state tree
/// which can be visualized using Graphviz tools or online viewers.
/// 
/// # Arguments
/// 
/// * `state` - A reference to the CFR state to export
/// * `output_path` - The path where the DOT file will be saved
/// 
/// # Returns
/// 
/// * `io::Result<()>` - Success or error
pub fn export_to_dot(state: &CFRState, output_path: &Path) -> io::Result<()> {
    let mut file = File::create(output_path)?;
    
    // Write DOT file header
    writeln!(file, "digraph CFRTree {{")?;
    
    // Configure graph styling
    writeln!(file, "  node [shape=box, style=\"rounded,filled\", fontname=\"Arial\"];")?;
    writeln!(file, "  edge [fontname=\"Arial\"];")?;
    
    // Access the internal state to get nodes
    let inner_state = state.internal_state();
    let nodes = &inner_state.borrow().nodes;
    
    // Process each node in the tree
    for node in nodes {
        // Node styling based on type
        let (color, shape) = match &node.data {
            NodeData::Root => ("lightblue", "doubleoctagon"),
            NodeData::Chance => ("lightgreen", "ellipse"),
            NodeData::Player(_) => ("coral", "box"),
            NodeData::Terminal(_) => ("lightgrey", "hexagon"),
        };
        
        // Node label content
        let label = match &node.data {
            NodeData::Root => format!("Root Node\\nIndex: {}", node.idx),
            NodeData::Chance => format!("Chance Node\\nIndex: {}", node.idx),
            NodeData::Player(_) => {
                // For player nodes, we need to determine which player is acting
                // We can infer the player from the node's position in the tree
                // Even positions after root are player 0, odd positions are player 1
                let player_seat = if node.idx % 2 == 1 { 0 } else { 1 };
                format!("Player {} Node\\nIndex: {}", player_seat, node.idx)
            },
            NodeData::Terminal(td) => format!("Terminal Node\\nIndex: {}\\nUtility: {:.2}", node.idx, td.total_utility),
        };
        
        // Write node with styling
        writeln!(
            file,
            "  node_{} [label=\"{}\", shape={}, style=\"filled\", fillcolor=\"{}\"];",
            node.idx, label, shape, color
        )?;
        
        // Process edges to children using the new iter_children method
        for (child_idx, child_node_idx) in node.iter_children() {
            let edge_label = match &node.data {
                NodeData::Chance => {
                    // For chance nodes, label the edge with the card
                    let card = Card::try_from(child_idx as u8).map_or_else(
                        |_| format!("Card {}", child_idx),
                        |card| format!("{}", card)
                    );
                    card
                },
                NodeData::Player(_) => {
                    // For player nodes, label the action
                    // Index 0 is fold according to the issue description
                    if child_idx == 0 {
                        "Fold".to_string()
                    } else if child_idx == 1 {
                        "Check/Call".to_string()
                    } else {
                        format!("Bet/Raise {}", child_idx - 1)
                    }
                },
                _ => format!("{}", child_idx),
            };
            
            // Use the count for this edge to indicate frequency
            let count = node.get_count(child_idx);
            let edge_style = if count > 0 {
                format!(" [label=\"{} (count: {})\", penwidth={}]", edge_label, count, 1.0 + (count as f32).min(5.0))
            } else {
                format!(" [label=\"{}\"]", edge_label)
            };
            
            writeln!(
                file,
                "  node_{} -> node_{}{}",
                node.idx, child_node_idx, edge_style
            )?;
        }
    }
    
    // Close the graph
    writeln!(file, "}}")?;
    
    Ok(())
}

/// Exports the CFR state tree to a PNG image using Graphviz.
/// 
/// This function creates a DOT file and then converts it to PNG using the 
/// Graphviz dot utility. The dot utility must be installed on the system.
/// 
/// # Arguments
/// 
/// * `state` - A reference to the CFR state to export
/// * `output_path` - The path where the PNG file will be saved
/// * `cleanup_dot` - Whether to remove the temporary DOT file after conversion
/// 
/// # Returns
/// 
/// * `io::Result<()>` - Success or error
pub fn export_to_png(state: &CFRState, output_path: &Path, cleanup_dot: bool) -> io::Result<()> {
    // Create a temporary DOT file
    let dot_path = output_path.with_extension("dot");
    
    // Generate DOT file
    export_to_dot(state, &dot_path)?;
    
    // Use Graphviz to convert DOT to PNG
    let status = std::process::Command::new("dot")
        .arg("-Tpng")
        .arg(&dot_path)
        .arg("-o")
        .arg(output_path)
        .status()?;
    
    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Failed to convert DOT to PNG. Make sure Graphviz is installed.",
        ));
    }
    
    // Clean up temporary DOT file if requested
    if cleanup_dot {
        std::fs::remove_file(dot_path)?;
    }
    
    Ok(())
}

/// Exports the CFR state tree to a SVG image.
/// 
/// This function creates a DOT file and then converts it to SVG using the 
/// Graphviz dot utility. The dot utility must be installed on the system.
/// SVG format is preferred for web viewing as it's scalable and interactive.
/// 
/// # Arguments
/// 
/// * `state` - A reference to the CFR state to export
/// * `output_path` - The path where the SVG file will be saved
/// * `cleanup_dot` - Whether to remove the temporary DOT file after conversion
/// 
/// # Returns
/// 
/// * `io::Result<()>` - Success or error
pub fn export_to_svg(state: &CFRState, output_path: &Path, cleanup_dot: bool) -> io::Result<()> {
    // Create a temporary DOT file
    let dot_path = output_path.with_extension("dot");
    
    // Generate DOT file
    export_to_dot(state, &dot_path)?;
    
    // Use Graphviz to convert DOT to SVG
    let status = std::process::Command::new("dot")
        .arg("-Tsvg")
        .arg(&dot_path)
        .arg("-o")
        .arg(output_path)
        .status()?;
    
    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Failed to convert DOT to SVG. Make sure Graphviz is installed.",
        ));
    }
    
    // Clean up temporary DOT file if requested
    if cleanup_dot {
        std::fs::remove_file(dot_path)?;
    }
    
    Ok(())
}

/// A convenience function that exports the CFR state to various formats.
/// 
/// # Arguments
/// 
/// * `state` - A reference to the CFR state to export
/// * `output_path` - The base path where the files will be saved
/// * `format` - The format(s) to export to: "dot", "png", "svg", or "all"
/// 
/// # Returns
/// 
/// * `io::Result<()>` - Success or error
pub fn export_cfr_state(state: &CFRState, output_path: &Path, format: &str) -> io::Result<()> {
    match format.to_lowercase().as_str() {
        "dot" => export_to_dot(state, output_path),
        "png" => export_to_png(state, output_path, true),
        "svg" => export_to_svg(state, output_path, true),
        "all" => {
            // For "all" format, we want to keep the DOT file
            export_to_dot(state, &output_path.with_extension("dot"))?;
            export_to_png(state, &output_path.with_extension("png"), false)?;
            export_to_svg(state, &output_path.with_extension("svg"), false)?;
            Ok(())
        },
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Unsupported format: {}. Use 'dot', 'png', 'svg', or 'all'.", format),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameState;
    use crate::arena::cfr::{CFRState, NodeData, PlayerData, TerminalData};
    use std::fs;
    
    /// Creates a standard CFR state with a variety of node types for testing.
    /// This represents a simple poker game tree with different possible paths.
    fn create_test_cfr_state() -> CFRState {
        // Create a game state with 2 players
        let game_state = GameState::new_starting(vec![100.0; 2], 10.0, 5.0, 0.0, 0);
        let mut cfr_state = CFRState::new(game_state);
        
        // Root -> Player 0 decision
        let player0_node = NodeData::Player(PlayerData { regret_matcher: None });
        let player0_idx = cfr_state.add(0, 0, player0_node);
        
        // Player 0 fold (idx 0 is fold according to issue)
        let terminal_fold = NodeData::Terminal(TerminalData::new(-10.0));
        let _fold_idx = cfr_state.add(player0_idx, 0, terminal_fold);
        
        // Player 0 call (idx 1)
        let player0_call = cfr_state.add(player0_idx, 1, NodeData::Chance);
        
        // Player 0 raise (idx 2)
        let player0_raise = cfr_state.add(player0_idx, 2, NodeData::Chance);
        
        // After call - chance node (dealing flop)
        for i in 0..3 {
            // Create 3 sample card possibilities (normally there would be more)
            let player1_node = NodeData::Player(PlayerData { regret_matcher: None });
            let player1_idx = cfr_state.add(player0_call, i, player1_node);
            
            // Player 1 fold
            let p1_fold_terminal = NodeData::Terminal(TerminalData::new(15.0));
            cfr_state.add(player1_idx, 0, p1_fold_terminal);
            
            // Player 1 call
            let p1_call_terminal = NodeData::Terminal(TerminalData::new(5.0));
            cfr_state.add(player1_idx, 1, p1_call_terminal);
        }
        
        // After raise - player 1 decision
        let player1_vs_raise = NodeData::Player(PlayerData { regret_matcher: None });
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
        if let Some(mut node) = cfr_state.get_mut(player0_idx) {
            node.increment_count(1);  // Call was taken once
            node.increment_count(2);  // Raise was taken twice
            node.increment_count(2);
        }
        
        cfr_state
    }
    
    #[test]
    fn test_export_to_dot_creates_file() {
        let cfr_state = create_test_cfr_state();
        
        // Create temp directory for test output
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("test_export.dot");
        
        // Export to DOT
        let result = export_to_dot(&cfr_state, &output_path);
        assert!(result.is_ok(), "Failed to export to DOT: {:?}", result.err());
        
        // Check that file exists
        assert!(output_path.exists(), "DOT file was not created");
        
        // Clean up
        temp_dir.close().unwrap();
    }
    
    #[test]
    fn test_different_node_types_displayed_correctly() {
        let cfr_state = create_test_cfr_state();
        
        // Create temp directory for test output
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("test_node_types.dot");
        
        // Export to DOT
        export_to_dot(&cfr_state, &output_path).unwrap();
        
        // Read the file content
        let content = fs::read_to_string(&output_path).unwrap();
        
        // Check for expected content for all node types
        // Root node
        assert!(content.contains("Root Node"), "Root node not properly labeled");
        assert!(content.contains("lightblue"), "Root node not properly colored");
        
        // Player nodes
        assert!(content.contains("Player 0") || content.contains("Player 1"), "Player node not properly labeled");
        assert!(content.contains("coral"), "Player node not properly colored");
        
        // Chance nodes
        assert!(content.contains("Chance Node"), "Chance node not properly labeled");
        assert!(content.contains("lightgreen"), "Chance node not properly colored");
        
        // Terminal nodes
        assert!(content.contains("Terminal Node"), "Terminal node not properly labeled");
        assert!(content.contains("Utility"), "Terminal node utility not displayed");
        assert!(content.contains("lightgrey"), "Terminal node not properly colored");
        
        // Edge labels
        assert!(content.contains("Fold"), "Fold action not properly labeled");
        assert!(content.contains("Check/Call"), "Call action not properly labeled");
        assert!(content.contains("Bet/Raise"), "Raise action not properly labeled");
        assert!(content.contains("count:"), "Action count not properly displayed");
        
        // Clean up
        temp_dir.close().unwrap();
    }
    
    #[test]
    fn test_export_creates_different_formats() {
        // Skip this test if graphviz is not installed
        if std::process::Command::new("dot").arg("-V").status().is_err() {
            println!("Skipping test_export_creates_different_formats - Graphviz not installed");
            return;
        }
        
        let cfr_state = create_test_cfr_state();
        
        // Create temp directory for test output
        let temp_dir = tempfile::tempdir().unwrap();
        
        // Test dot format
        let dot_path = temp_dir.path().join("test.dot");
        let dot_result = export_to_dot(&cfr_state, &dot_path);
        assert!(dot_result.is_ok(), "DOT export failed: {:?}", dot_result.err());
        assert!(dot_path.exists(), "DOT file was not created");
        
        // Test png format (requires graphviz)
        let png_path = temp_dir.path().join("test.png");
        let png_result = export_to_png(&cfr_state, &png_path, true);
        assert!(png_result.is_ok(), "PNG export failed: {:?}", png_result.err());
        assert!(png_path.exists(), "PNG file was not created");
        
        // Test svg format (requires graphviz)
        let svg_path = temp_dir.path().join("test.svg");
        let svg_result = export_to_svg(&cfr_state, &svg_path, true);
        assert!(svg_result.is_ok(), "SVG export failed: {:?}", svg_result.err());
        assert!(svg_path.exists(), "SVG file was not created");
        
        // Test the convenience function with "all" format
        let all_base_path = temp_dir.path().join("test_all");
        let all_result = export_cfr_state(&cfr_state, &all_base_path, "all");
        assert!(all_result.is_ok(), "All formats export failed: {:?}", all_result.err());
        
        // Check that all three formats were created
        let all_dot_path = all_base_path.with_extension("dot");
        let all_png_path = all_base_path.with_extension("png");
        let all_svg_path = all_base_path.with_extension("svg");
        
        assert!(all_dot_path.exists(), "DOT file not created in 'all' format at {:?}", all_dot_path);
        assert!(all_png_path.exists(), "PNG file not created in 'all' format at {:?}", all_png_path);
        assert!(all_svg_path.exists(), "SVG file not created in 'all' format at {:?}", all_svg_path);
        
        // Print file contents for debugging if they don't exist
        if !all_dot_path.exists() || !all_png_path.exists() || !all_svg_path.exists() {
            println!("Directory contents:");
            if let Ok(entries) = std::fs::read_dir(temp_dir.path()) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        println!("  {:?}", entry.path());
                    }
                }
            }
        }
        
        // Clean up
        temp_dir.close().unwrap();
    }
    
    #[test]
    fn test_invalid_format_returns_error() {
        let cfr_state = create_test_cfr_state();
        let temp_dir = tempfile::tempdir().unwrap();
        let invalid_path = temp_dir.path().join("invalid_format");
        
        let result = export_cfr_state(&cfr_state, &invalid_path, "invalid_format");
        assert!(result.is_err());
        
        // Clean up
        temp_dir.close().unwrap();
    }
    
    #[test]
    fn test_player_seat_labeling() {
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("player_seats.dot");
        
        // Create a test CFR state with multiple player nodes
        let game_state = GameState::new_starting(vec![100.0; 2], 10.0, 5.0, 0.0, 0);
        let mut cfr_state = CFRState::new(game_state);
        
        // Add player nodes at different positions
        let player0_node = NodeData::Player(PlayerData { regret_matcher: None });
        let player0_idx = cfr_state.add(0, 0, player0_node.clone());
        
        let player1_node = NodeData::Player(PlayerData { regret_matcher: None });
        let player1_idx = cfr_state.add(player0_idx, 1, player1_node);
        
        // Export to DOT format
        export_to_dot(&cfr_state, &output_path).unwrap();
        
        // Read the generated file
        let dot_content = fs::read_to_string(&output_path).unwrap();
        
        // Verify player seat labels are present
        assert!(dot_content.contains("Player 0 Node"));
        assert!(dot_content.contains("Player 1 Node"));
    }
}
