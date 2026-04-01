use std::collections::HashMap;

use ratatui::{
    Frame,
    layout::{Constraint, Rect},
    style::{Color, Style},
    widgets::{Cell, Row, Table, TableState},
};

use crate::tui::{
    state::GameLogEntry,
    theme::{ICON_GAMES, OVERLAY0, SURFACE1, TEXT, header_style, panel_block, profit_style},
};

fn entry_to_row(entry: &GameLogEntry, agent_colors: &HashMap<&str, Color>) -> Row<'static> {
    let winner_color = agent_colors
        .get(entry.winner_name.as_str())
        .copied()
        .unwrap_or(TEXT);
    let loser_color = agent_colors
        .get(entry.loser_name.as_str())
        .copied()
        .unwrap_or(TEXT);
    let cells = vec![
        Cell::from(format!("{}", entry.game_number)).style(Style::default().fg(OVERLAY0)),
        Cell::from(entry.winner_name.clone()).style(Style::default().fg(winner_color)),
        Cell::from(format!("{:+.0}", entry.winner_profit)).style(profit_style(entry.winner_profit)),
        Cell::from(entry.loser_name.clone()).style(Style::default().fg(loser_color)),
        Cell::from(format!("{:+.0}", entry.loser_loss)).style(profit_style(entry.loser_loss)),
        Cell::from(format!("{:.0}", entry.pot_size)).style(Style::default().fg(TEXT)),
        Cell::from(format!("{}", entry.ending_round)).style(Style::default().fg(OVERLAY0)),
    ];
    Row::new(cells)
}

pub fn render_game_log(
    frame: &mut Frame,
    area: Rect,
    log: &[GameLogEntry],
    scroll: usize,
    selected: Option<usize>,
    focused: bool,
    agent_colors: &HashMap<&str, Color>,
) {
    // Virtualize: only build Rows for the visible window.
    // The block border + title + header consume rows.
    let visible_rows = area.height.saturating_sub(5) as usize;
    let window_end = (scroll + visible_rows + 1).min(log.len());
    let window_start = scroll.min(window_end);
    let window = &log[window_start..window_end];

    let header_cells = vec![
        Cell::from("#").style(header_style()),
        Cell::from("Winner").style(header_style()),
        Cell::from("Win").style(header_style()),
        Cell::from("Loser").style(header_style()),
        Cell::from("Loss").style(header_style()),
        Cell::from("Pot").style(header_style()),
        Cell::from("Street").style(header_style()),
    ];
    let header = Row::new(header_cells).height(1).bottom_margin(1);

    let rows: Vec<Row> = window
        .iter()
        .map(|entry| entry_to_row(entry, agent_colors))
        .collect();

    let widths = [
        Constraint::Length(7),
        Constraint::Min(10),
        Constraint::Length(8),
        Constraint::Min(10),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(9),
    ];

    // Adjust selected to be relative to the window start
    let relative_selected = selected.and_then(|s| s.checked_sub(window_start));

    let title = format!("{} Recent Games", ICON_GAMES);
    let block = panel_block(&title, focused);

    let table = Table::new(rows, widths)
        .header(header)
        .block(block)
        .row_highlight_style(Style::default().bg(SURFACE1).fg(TEXT));

    let mut state = TableState::default();
    state.select(relative_selected);

    frame.render_stateful_widget(table, area, &mut state);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::state::RoundLabel;
    use crate::tui::theme;
    use insta::assert_snapshot;
    use ratatui::{Terminal, backend::TestBackend};

    fn test_agent_colors<'a>(names: &'a [&'a str]) -> HashMap<&'a str, Color> {
        names
            .iter()
            .enumerate()
            .map(|(i, &n)| (n, theme::agent_color(i)))
            .collect()
    }

    #[test]
    fn test_render_game_log() {
        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let entries = [
            GameLogEntry::new(
                1,
                vec!["Alice".into(), "Bob".into()],
                vec![15.0, -15.0],
                RoundLabel::River,
                10.0,
            ),
            GameLogEntry::new(
                2,
                vec!["Alice".into(), "Bob".into()],
                vec![-5.0, 5.0],
                RoundLabel::Flop,
                10.0,
            ),
        ];
        let colors = test_agent_colors(&["Alice", "Bob"]);
        terminal
            .draw(|frame| {
                render_game_log(frame, frame.area(), &entries, 0, None, true, &colors);
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }

    #[test]
    fn test_render_game_log_empty() {
        let backend = TestBackend::new(80, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        let colors = HashMap::new();
        terminal
            .draw(|frame| {
                render_game_log(frame, frame.area(), &[], 0, None, false, &colors);
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }

    #[test]
    fn test_virtualized_only_builds_visible_items() {
        // Create 10000 entries but render in a 12-row area (10 visible rows).
        // This should NOT build 10000 Rows.
        let entries: Vec<GameLogEntry> = (1..=10_000)
            .map(|i| GameLogEntry::new(i, vec!["A".into()], vec![1.0], RoundLabel::Preflop, 10.0))
            .collect();

        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let colors = test_agent_colors(&["A"]);

        // Scroll to middle (game 5000)
        terminal
            .draw(|frame| {
                render_game_log(
                    frame,
                    frame.area(),
                    &entries,
                    5000,
                    Some(5000),
                    true,
                    &colors,
                );
            })
            .unwrap();

        // Verify the buffer contains game #5001 (scroll=5000 is 0-indexed)
        let buf = terminal.backend().buffer().clone();
        let content = (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            content.contains("5001"),
            "Should show game #5001 at scroll=5000"
        );
        assert!(
            !content.contains("#1 "),
            "Should NOT show game #1 when scrolled to 5000"
        );
    }
}
