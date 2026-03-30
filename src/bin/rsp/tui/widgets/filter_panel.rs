use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{List, ListItem, ListState},
};

use crate::tui::{
    state::{FilterState, RoundLabel},
    theme::{
        CHECK_OFF, CHECK_ON, GREEN, ICON_FILTER, MAUVE, OVERLAY1, PEACH, RED, SKY, SURFACE1, TEXT,
        header_style, panel_block,
    },
};

/// The kinds of items in the filter panel list.
#[derive(Debug, Clone)]
pub enum FilterItem {
    /// Section header (not toggleable).
    Header(String),
    /// A winner agent filter.
    Winner(String),
    /// A participant agent filter.
    Participant(String),
    /// A street filter.
    Street(RoundLabel),
    /// A player count filter.
    PlayerCount(usize),
    /// Clear all filters action.
    ClearAll,
}

/// Build the flat list of filter items from current agent names and player counts.
pub fn build_filter_items(agent_names: &[String], player_counts: &[usize]) -> Vec<FilterItem> {
    let mut items = Vec::new();

    items.push(FilterItem::Header("Winner".into()));
    for name in agent_names {
        items.push(FilterItem::Winner(name.clone()));
    }

    items.push(FilterItem::Header("Participant".into()));
    for name in agent_names {
        items.push(FilterItem::Participant(name.clone()));
    }

    items.push(FilterItem::Header("Street".into()));
    items.push(FilterItem::Street(RoundLabel::Preflop));
    items.push(FilterItem::Street(RoundLabel::Flop));
    items.push(FilterItem::Street(RoundLabel::Turn));
    items.push(FilterItem::Street(RoundLabel::River));
    items.push(FilterItem::Street(RoundLabel::Showdown));

    if !player_counts.is_empty() {
        items.push(FilterItem::Header("Players".into()));
        for &count in player_counts {
            items.push(FilterItem::PlayerCount(count));
        }
    }

    items.push(FilterItem::ClearAll);

    items
}

/// Render the filter panel widget.
pub fn render_filter_panel(
    frame: &mut Frame,
    area: Rect,
    filter: &FilterState,
    agent_names: &[String],
    player_counts: &[usize],
    focused: bool,
) {
    let items = build_filter_items(agent_names, player_counts);

    let list_items: Vec<ListItem> = items
        .iter()
        .map(|item| match item {
            FilterItem::Header(label) => ListItem::new(Line::from(Span::styled(
                format!("─── {} ───", label),
                header_style(),
            ))),
            FilterItem::Winner(name) => {
                let checked = filter.winners.contains(name);
                let (symbol, color) = if checked {
                    (CHECK_ON, GREEN)
                } else {
                    (CHECK_OFF, OVERLAY1)
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {} ", symbol), Style::default().fg(color)),
                    Span::raw(name.as_str()),
                ]))
            }
            FilterItem::Participant(name) => {
                let checked = filter.participants.contains(name);
                let (symbol, color) = if checked {
                    (CHECK_ON, SKY)
                } else {
                    (CHECK_OFF, OVERLAY1)
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {} ", symbol), Style::default().fg(color)),
                    Span::raw(name.as_str()),
                ]))
            }
            FilterItem::Street(round) => {
                let checked = filter.streets.contains(round);
                let (symbol, color) = if checked {
                    (CHECK_ON, MAUVE)
                } else {
                    (CHECK_OFF, OVERLAY1)
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {} ", symbol), Style::default().fg(color)),
                    Span::raw(format!("{}", round)),
                ]))
            }
            FilterItem::PlayerCount(count) => {
                let checked = filter.player_counts.contains(count);
                let (symbol, color) = if checked {
                    (CHECK_ON, PEACH)
                } else {
                    (CHECK_OFF, OVERLAY1)
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {} ", symbol), Style::default().fg(color)),
                    Span::raw(format!("{}-player", count)),
                ]))
            }
            FilterItem::ClearAll => ListItem::new(Line::from(Span::styled(
                "  ✕ Clear All",
                Style::default().fg(RED).add_modifier(Modifier::BOLD),
            ))),
        })
        .collect();

    let title = format!("{} Filters", ICON_FILTER);
    let block = panel_block(&title, focused);
    let list = List::new(list_items)
        .block(block)
        .highlight_style(Style::default().bg(SURFACE1).fg(TEXT));

    let mut state = ListState::default();
    state.select(Some(filter.selected));

    frame.render_stateful_widget(list, area, &mut state);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_filter_items_structure() {
        let names = vec!["Alice".into(), "Bob".into()];
        let items = build_filter_items(&names, &[2, 6]);

        // Header "Winner" + 2 winners + Header "Participant" + 2 participants
        // + Header "Street" + 5 streets + Header "Players" + 2 counts + ClearAll = 16
        assert_eq!(items.len(), 16);

        assert!(matches!(&items[0], FilterItem::Header(s) if s == "Winner"));
        assert!(matches!(&items[1], FilterItem::Winner(s) if s == "Alice"));
        assert!(matches!(&items[2], FilterItem::Winner(s) if s == "Bob"));
        assert!(matches!(&items[3], FilterItem::Header(s) if s == "Participant"));
        assert!(matches!(&items[4], FilterItem::Participant(s) if s == "Alice"));
        assert!(matches!(&items[5], FilterItem::Participant(s) if s == "Bob"));
        assert!(matches!(&items[6], FilterItem::Header(s) if s == "Street"));
        assert!(matches!(&items[7], FilterItem::Street(RoundLabel::Preflop)));
        assert!(matches!(&items[12], FilterItem::Header(s) if s == "Players"));
        assert!(matches!(&items[13], FilterItem::PlayerCount(2)));
        assert!(matches!(&items[14], FilterItem::PlayerCount(6)));
        assert!(matches!(&items[15], FilterItem::ClearAll));
    }

    #[test]
    fn test_build_filter_items_empty_agents() {
        let items = build_filter_items(&[], &[]);
        // 3 headers + 5 streets + ClearAll = 9
        assert_eq!(items.len(), 9);
    }

    #[test]
    fn test_build_filter_items_no_player_counts() {
        let names = vec!["Alice".into()];
        let items = build_filter_items(&names, &[]);
        // No "Players" header when no counts observed
        assert!(
            !items
                .iter()
                .any(|i| matches!(i, FilterItem::Header(s) if s == "Players"))
        );
    }
}
