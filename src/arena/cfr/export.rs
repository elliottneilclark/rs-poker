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
//! ```

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::str::FromStr;
use std::process::Command;

use crate::core::Card;

use super::{CFRState, NodeData};

/// The default font used for node and edge labels in the graph visualization.
const DEFAULT_FONT: &str = "Arial";

/// Color scheme for different node types
const COLOR_ROOT: &str = "lightblue";     // Light blue for root nodes
const COLOR_CHANCE: &str = "lightgreen";   // Light green for chance nodes
const COLOR_PLAYER: &str = "coral";        // Coral for player nodes
const COLOR_TERMINAL: &str = "lightgrey";  // Light grey for terminal nodes

/// The format to export the CFR state tree to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Exports to DOT (Graphviz) format
    Dot,
    /// Exports to PNG format
    Png,
    /// Exports to SVG format
    Svg,
    /// Exports to all available formats
    All,
}

impl FromStr for ExportFormat {
    type Err = io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "dot" => Ok(ExportFormat::Dot),
            "png" => Ok(ExportFormat::Png),
            "svg" => Ok(ExportFormat::Svg),
            "all" => Ok(ExportFormat::All),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid export format: {}. Valid formats are: dot, png, svg, all", s),
            )),
        }
    }
}

/// Generates a DOT format representation of the CFR state tree.
/// 
/// This function creates a DOT representation of the CFR state tree in memory,
/// which can be used for visualization or testing.
/// 
/// # Arguments
/// 
/// * `state` - A reference to the CFR state to convert to DOT format
/// 
/// # Returns
/// 
/// * `String` - The DOT format representation of the tree
pub fn generate_dot(state: &CFRState) -> String {
    let mut output = String::new();
    
    // Write DOT file header with enhanced styling
    output.push_str("digraph CFRTree {\n");
    output.push_str("  // Graph styling\n");
    output.push_str("  graph [rankdir=TB, splines=polyline, nodesep=1.0, ranksep=1.2, concentrate=true, compound=true];\n");
    output.push_str(&format!("  node [shape=box, style=\"rounded,filled\", fontname=\"{}\", margin=0.2];\n", DEFAULT_FONT));
    output.push_str(&format!("  edge [fontname=\"{}\", penwidth=1.0, labelangle=25, labeldistance=1.8, labelfloat=true];\n", DEFAULT_FONT));
    
    // Add legend as a separate subgraph with explicit positioning
    output.push_str("  // Add legend\n");
    output.push_str("  subgraph cluster_legend {\n");
    output.push_str("    graph [rank=sink];\n");
    output.push_str("    label=\"Legend\";\n");
    output.push_str("    style=rounded;\n");
    output.push_str("    color=gray;\n");
    output.push_str("    margin=16;\n");
    output.push_str("    node [shape=plaintext, style=\"\"];\n");
    output.push_str("    legend [label=<\n");
    output.push_str("      <table border=\"0\" cellborder=\"0\" cellspacing=\"2\">\n");
    output.push_str("        <tr><td align=\"left\"><b>Node Types:</b></td></tr>\n");
    output.push_str("        <tr><td align=\"left\">• Root (⬢): Light Blue - Starting state</td></tr>\n");
    output.push_str("        <tr><td align=\"left\">• Player (□): Coral - Decision points</td></tr>\n");
    output.push_str("        <tr><td align=\"left\">• Chance (○): Light Green - Card deals</td></tr>\n");
    output.push_str("        <tr><td align=\"left\">• Terminal (⬡): Light Grey - Final states</td></tr>\n");
    output.push_str("        <tr><td><br/></td></tr>\n");
    output.push_str("        <tr><td align=\"left\"><b>Edge Properties:</b></td></tr>\n");
    output.push_str("        <tr><td align=\"left\">• Thickness: Usage frequency</td></tr>\n");
    output.push_str("        <tr><td align=\"left\">• Labels: Action/Card</td></tr>\n");
    output.push_str("        <tr><td align=\"left\">• Percent: Visit frequency</td></tr>\n");
    output.push_str("      </table>\n");
    output.push_str("    >];\n");
    output.push_str("  }\n\n");
    
    // Add root node at the top
    output.push_str("  // Node grouping\n");
    output.push_str("  {rank=source; node_0;}\n");
    
    let inner_state = state.internal_state();
    let nodes = &inner_state.borrow().nodes;
    
    // Process nodes
    for node in nodes {
        let (color, shape, style) = match &node.data {
            NodeData::Root => (COLOR_ROOT, "doubleoctagon", "filled"),
            NodeData::Chance => (COLOR_CHANCE, "ellipse", "filled"),
            NodeData::Player(_) => (COLOR_PLAYER, "box", "rounded,filled"),
            NodeData::Terminal(_) => (COLOR_TERMINAL, "hexagon", "filled"),
        };
        
        let total_visits: u32 = (0..52)
            .map(|i| node.get_count(i))
            .sum();
        
        let label = match &node.data {
            NodeData::Root => format!(
                "Root Node\\nIndex: {}\\nTotal Visits: {}", 
                node.idx, total_visits
            ),
            NodeData::Chance => format!(
                "Chance Node\\nIndex: {}\\nTotal Visits: {}", 
                node.idx, total_visits
            ),
            NodeData::Player(_) => {
                let player_seat = if node.idx % 2 == 1 { 0 } else { 1 };
                format!(
                    "Player {} Node\\nIndex: {}\\nTotal Visits: {}",
                    player_seat, node.idx, total_visits
                )
            },
            NodeData::Terminal(td) => format!(
                "Terminal Node\\nIndex: {}\\nUtility: {:.2}\\nVisits: {}",
                node.idx, td.total_utility, total_visits
            ),
        };
        
        let tooltip = match &node.data {
            NodeData::Terminal(td) => format!(
                "Average Utility: {:.2}",
                if total_visits > 0 { td.total_utility / total_visits as f32 } else { 0.0 }
            ),
            _ => {
                let (most_common_idx, most_common_count) = (0..52)
                    .map(|i| (i, node.get_count(i)))
                    .max_by_key(|&(_, count)| count)
                    .unwrap_or((0, 0));
                format!(
                    "Most Common Action: {}\\nAction Frequency: {:.1}%",
                    most_common_idx,
                    if total_visits > 0 { (most_common_count as f32 / total_visits as f32) * 100.0 } else { 0.0 }
                )
            },
        };
        
        output.push_str(&format!(
            "  node_{} [label=\"{}\", shape={}, style=\"{}\", fillcolor=\"{}\", tooltip=\"{}\"];\n",
            node.idx, label, shape, style, color, tooltip
        ));
        
        let total_count: u32 = (0..52)
            .map(|i| node.get_count(i))
            .sum();
        
        // Group nodes by level for better layout
        if let NodeData::Player(_) = node.data {
            output.push_str(&format!("  {{rank=same; node_{};}}  // Group player nodes\n", node.idx));
        }
        
        for (child_idx, child_node_idx) in node.iter_children() {
            let edge_label = match &node.data {
                NodeData::Chance => {
                    let card = Card::try_from(child_idx as u8).map_or_else(
                        |_| format!("Card {}", child_idx),
                        |card| format!("{}", card)
                    );
                    card
                },
                NodeData::Player(_) => {
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
            
            let count = node.get_count(child_idx);
            let edge_style = if total_count > 0 {
                let percentage = (count as f32 / total_count as f32) * 100.0;
                let penwidth = 1.0 + (percentage / 10.0).min(8.0);
                let color = format!("#{:02X}{:02X}FF", 
                    (155.0 + percentage).min(255.0) as u8,
                    (155.0 + percentage).min(255.0) as u8
                );
                format!(
                    " [label=\"{}\", penwidth={}, color=\"{}\", tooltip=\"Frequency: {:.1}%\", xlabel=\"{:.0}%\", weight={}]",
                    edge_label, penwidth, color, percentage, percentage,
                    if percentage > 0.0 { percentage as u32 } else { 1 }
                )
            } else {
                format!(" [label=\"{}\", weight=1]", edge_label)
            };
            
            output.push_str(&format!(
                "  node_{} -> node_{}{}\n",
                node.idx, child_node_idx, edge_style
            ));
        }
    }
    
    output.push_str("}\n");
    output
}

/// Exports the CFR state tree to a DOT (Graphviz) file.
/// 
/// This function writes the DOT representation of the CFR state tree to a file,
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
    let dot_content = generate_dot(state);
    let mut file = File::create(output_path)?;
    file.write_all(dot_content.as_bytes())
}

/// Helper function to convert DOT file to another format using Graphviz.
/// 
/// # Arguments
/// 
/// * `dot_path` - Path to the source DOT file
/// * `output_path` - Path where the converted file will be saved
/// * `format` - The target format (e.g., "png", "svg")
/// * `cleanup_dot` - Whether to remove the DOT file after conversion
/// 
/// # Returns
/// 
/// * `io::Result<()>` - Success or error
fn convert_with_graphviz(dot_path: &Path, output_path: &Path, format: &str, cleanup_dot: bool) -> io::Result<()> {
    // Use Graphviz to convert DOT to target format
    let status = Command::new("dot")
        .arg(format!("-T{}", format))
        .arg(dot_path)
        .arg("-o")
        .arg(output_path)
        .status()?;

    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to convert DOT to {}. Make sure Graphviz is installed.", format.to_uppercase()),
        ));
    }

    // Clean up temporary DOT file if requested
    if cleanup_dot {
        std::fs::remove_file(dot_path)?;
    }

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
    let dot_path = output_path.with_extension("dot");
    export_to_dot(state, &dot_path)?;
    convert_with_graphviz(&dot_path, output_path, "png", cleanup_dot)
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
    let dot_path = output_path.with_extension("dot");
    export_to_dot(state, &dot_path)?;
    convert_with_graphviz(&dot_path, output_path, "svg", cleanup_dot)
}

/// A convenience function that exports the CFR state to various formats.
/// 
/// # Arguments
/// 
/// * `state` - A reference to the CFR state to export
/// * `output_path` - The base path where the files will be saved
/// * `format` - The format to export to
/// 
/// # Returns
/// 
/// * `io::Result<()>` - Success or error
pub fn export_cfr_state(state: &CFRState, output_path: &Path, format: ExportFormat) -> io::Result<()> {
    match format {
        ExportFormat::Dot => export_to_dot(state, output_path),
        ExportFormat::Png => export_to_png(state, output_path, true),
        ExportFormat::Svg => export_to_svg(state, output_path, true),
        ExportFormat::All => {
            // For "all" format, we want to keep the DOT file
            let dot_path = output_path.with_extension("dot");
            let png_path = output_path.with_extension("png");
            let svg_path = output_path.with_extension("svg");
            
            export_to_dot(state, &dot_path)?;
            convert_with_graphviz(&dot_path, &png_path, "png", false)?;
            convert_with_graphviz(&dot_path, &svg_path, "svg", false)?;
            Ok(())
        }
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
        assert!(content.contains("%"), "Action percentages not properly displayed");
        
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
        let all_result = export_cfr_state(&cfr_state, &all_base_path, ExportFormat::All);
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
        let _cfr_state = create_test_cfr_state();
        let temp_dir = tempfile::tempdir().unwrap();
        let _invalid_path = temp_dir.path().join("invalid_format");
        
        // Test with an invalid format string
        let result = ExportFormat::from_str("invalid_format");
        assert!(result.is_err(), "Should error on invalid format string");
        
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid export format"), 
                "Error message should mention invalid format");
            assert!(e.to_string().contains("Valid formats are:"), 
                "Error message should list valid formats");
        }
        
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
        let _player1_idx = cfr_state.add(player0_idx, 1, player1_node);
        
        // Export to DOT format
        export_to_dot(&cfr_state, &output_path).unwrap();
        
        // Read the generated file
        let dot_content = fs::read_to_string(&output_path).unwrap();
        
        // Verify player seat labels are present
        assert!(dot_content.contains("Player 0 Node"));
        assert!(dot_content.contains("Player 1 Node"));
    }

    #[test]
    fn test_generate_dot_output() {
        let cfr_state = create_test_cfr_state();
        let dot_content = generate_dot(&cfr_state);

        // Print the DOT content for debugging
        println!("Generated DOT content:\n{}", dot_content);

        // Basic structure checks
        assert!(dot_content.starts_with("digraph CFRTree {"), "Missing graph header");
        assert!(dot_content.ends_with("}\n"), "Missing graph closing");
        
        // Font settings
        assert!(dot_content.contains(&format!("fontname=\"{}\"", DEFAULT_FONT)), "Missing font settings");
        
        // Node type checks
        assert!(dot_content.contains("fillcolor=\"lightblue\""), "Missing root node style");
        assert!(dot_content.contains("fillcolor=\"lightgreen\""), "Missing chance node style");
        assert!(dot_content.contains("fillcolor=\"coral\""), "Missing player node style");
        assert!(dot_content.contains("fillcolor=\"lightgrey\""), "Missing terminal node style");
        
        // Node content checks
        assert!(dot_content.contains("Root Node"), "Missing root node label");
        assert!(dot_content.contains("Player 0 Node"), "Missing player 0 label");
        assert!(dot_content.contains("Player 1 Node"), "Missing player 1 label");
        assert!(dot_content.contains("Terminal Node"), "Missing terminal node label");
        assert!(dot_content.contains("Utility:"), "Missing utility value");
        
        // Edge label checks
        assert!(dot_content.contains("label=\"Fold\""), "Missing fold action label");
        assert!(dot_content.contains("label=\"Check/Call\""), "Missing call action label");
        assert!(dot_content.contains("xlabel=\"33%\""), "Missing call percentage");
        assert!(dot_content.contains("label=\"Bet/Raise 1\""), "Missing raise action label");
        assert!(dot_content.contains("xlabel=\"67%\""), "Missing raise percentage");
        
        // Verify edge thickness
        assert!(dot_content.contains("penwidth=4.333"), "Edge thickness for call (33.3%) not correct");
        assert!(dot_content.contains("penwidth=7.666"), "Edge thickness for raise (66.7%) not correct");
    }

    #[test]
    fn test_dot_generation_matches_file_output() {
        let cfr_state = create_test_cfr_state();
        let dot_content = generate_dot(&cfr_state);
        
        // Create temp directory for test output
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("test_match.dot");
        
        // Export to file
        export_to_dot(&cfr_state, &output_path).unwrap();
        
        // Read the file content
        let file_content = fs::read_to_string(&output_path).unwrap();
        
        // Compare in-memory generation with file output
        assert_eq!(dot_content, file_content, "Generated DOT content should match file output");
        
        // Clean up
        temp_dir.close().unwrap();
    }
}
