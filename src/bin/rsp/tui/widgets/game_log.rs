use ratatui::{
    Frame,
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{List, ListItem, ListState},
};

use crate::tui::{
    state::GameLogEntry,
    theme::{ICON_GAMES, OVERLAY0, SURFACE1, TEXT, panel_block, profit_style},
};

fn entry_to_list_item(entry: &GameLogEntry) -> ListItem<'static> {
    let mut spans = vec![Span::styled(
        format!("#{:<5} ", entry.game_number),
        Style::default().fg(OVERLAY0),
    )];

    for (name, profit) in entry.agent_names.iter().zip(entry.profits.iter()) {
        spans.push(Span::raw(format!("{} ", name)));
        spans.push(Span::styled(
            format!("{:+.0} ", profit),
            profit_style(*profit),
        ));
    }

    spans.push(Span::styled(
        format!("({})", entry.ending_round),
        Style::default().fg(OVERLAY0),
    ));

    ListItem::new(Line::from(spans))
}

pub fn render_game_log(
    frame: &mut Frame,
    area: Rect,
    log: &[GameLogEntry],
    scroll: usize,
    selected: Option<usize>,
    focused: bool,
) {
    // Virtualize: only build ListItems for the visible window.
    // The block border + title consume 2 rows.
    let visible_rows = area.height.saturating_sub(2) as usize;
    let window_end = (scroll + visible_rows + 1).min(log.len());
    let window_start = scroll.min(window_end);
    let window = &log[window_start..window_end];

    let items: Vec<ListItem> = window
        .iter()
        .map(|entry| entry_to_list_item(entry))
        .collect();

    // Adjust selected to be relative to the window start
    let relative_selected = selected.and_then(|s| s.checked_sub(window_start));

    let title = format!("{} Recent Games", ICON_GAMES);
    let block = panel_block(&title, focused);

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().bg(SURFACE1).fg(TEXT));

    let mut state = ListState::default();
    state.select(relative_selected);

    frame.render_stateful_widget(list, area, &mut state);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::state::RoundLabel;
    use insta::assert_snapshot;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn test_render_game_log() {
        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let entries = [
            GameLogEntry {
                game_number: 1,
                agent_names: vec!["Alice".into(), "Bob".into()],
                profits: vec![15.0, -15.0],
                ending_round: RoundLabel::River,
            },
            GameLogEntry {
                game_number: 2,
                agent_names: vec!["Alice".into(), "Bob".into()],
                profits: vec![-5.0, 5.0],
                ending_round: RoundLabel::Flop,
            },
        ];
        terminal
            .draw(|frame| {
                render_game_log(frame, frame.area(), &entries, 0, None, true);
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }

    #[test]
    fn test_render_game_log_empty() {
        let backend = TestBackend::new(80, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render_game_log(frame, frame.area(), &[], 0, None, false);
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }

    #[test]
    fn test_virtualized_only_builds_visible_items() {
        // Create 10000 entries but render in a 12-row area (10 visible rows).
        // This should NOT build 10000 ListItems.
        let entries: Vec<GameLogEntry> = (1..=10_000)
            .map(|i| GameLogEntry {
                game_number: i,
                agent_names: vec!["A".into()],
                profits: vec![1.0],
                ending_round: RoundLabel::Preflop,
            })
            .collect();

        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::new(backend).unwrap();

        // Scroll to middle (game 5000)
        terminal
            .draw(|frame| {
                render_game_log(frame, frame.area(), &entries, 5000, Some(5000), true);
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
            content.contains("#5001"),
            "Should show game #5001 at scroll=5000"
        );
        assert!(
            !content.contains("#1 "),
            "Should NOT show game #1 when scrolled to 5000"
        );
    }
}
