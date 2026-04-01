use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{List, ListItem, ListState},
};

use crate::tui::{
    state::{ALL_PROFIT_BUCKETS, FilterState, ProfitBucket, RoundLabel},
    theme::{
        CHECK_OFF, CHECK_ON, FLAMINGO, GREEN, ICON_FILTER, MAUVE, OVERLAY1, PEACH, RED, SKY,
        SURFACE1, TEXT, header_style, panel_block,
    },
};

/// The kinds of items in the filter panel list.
#[derive(Debug, Clone)]
pub enum FilterItem {
    /// Section header (not toggleable).
    Header(String),
    /// A winner agent filter.
    Winner(String),
    /// A loser (biggest loser) filter.
    Loser(String),
    /// A participant agent filter.
    Participant(String),
    /// A street filter.
    Street(RoundLabel),
    /// A win size bucket filter.
    WinSize(ProfitBucket),
    /// A loss size bucket filter.
    LossSize(ProfitBucket),
    /// A player count filter.
    PlayerCount(usize),
    /// Clear all filters action.
    ClearAll,
}

/// Build a checkbox list item with consistent styling.
fn checkbox_item(label: &str, checked: bool, active_color: Color) -> ListItem<'static> {
    let (symbol, color) = if checked {
        (CHECK_ON, active_color)
    } else {
        (CHECK_OFF, OVERLAY1)
    };
    ListItem::new(Line::from(vec![
        Span::styled(format!(" {} ", symbol), Style::default().fg(color)),
        Span::raw(label.to_string()),
    ]))
}

/// Build the flat list of filter items from current agent names and player counts.
pub fn build_filter_items(agent_names: &[String], player_counts: &[usize]) -> Vec<FilterItem> {
    let mut items = Vec::new();

    items.push(FilterItem::Header("Winner".into()));
    for name in agent_names {
        items.push(FilterItem::Winner(name.clone()));
    }

    items.push(FilterItem::Header("Loser".into()));
    for name in agent_names {
        items.push(FilterItem::Loser(name.clone()));
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

    items.push(FilterItem::Header("Win Size".into()));
    for &bucket in &ALL_PROFIT_BUCKETS {
        items.push(FilterItem::WinSize(bucket));
    }

    items.push(FilterItem::Header("Loss Size".into()));
    for &bucket in &ALL_PROFIT_BUCKETS {
        items.push(FilterItem::LossSize(bucket));
    }

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
            FilterItem::Winner(name) => checkbox_item(name, filter.winners.contains(name), GREEN),
            FilterItem::Loser(name) => checkbox_item(name, filter.losers.contains(name), RED),
            FilterItem::Participant(name) => {
                checkbox_item(name, filter.participants.contains(name), SKY)
            }
            FilterItem::Street(round) => {
                checkbox_item(&format!("{}", round), filter.streets.contains(round), MAUVE)
            }
            FilterItem::WinSize(bucket) => {
                checkbox_item(bucket.label(), filter.win_sizes.contains(bucket), GREEN)
            }
            FilterItem::LossSize(bucket) => {
                checkbox_item(bucket.label(), filter.loss_sizes.contains(bucket), FLAMINGO)
            }
            FilterItem::PlayerCount(count) => checkbox_item(
                &format!("{}-player", count),
                filter.player_counts.contains(count),
                PEACH,
            ),
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

        // Header "Winner" + 2 winners
        // + Header "Loser" + 2 losers
        // + Header "Participant" + 2 participants
        // + Header "Street" + 5 streets
        // + Header "Win Size" + 4 buckets
        // + Header "Loss Size" + 4 buckets
        // + Header "Players" + 2 counts
        // + ClearAll
        // = 7 headers + 2+2+2+5+4+4+2 items + 1 ClearAll = 29
        assert_eq!(items.len(), 29);

        assert!(matches!(&items[0], FilterItem::Header(s) if s == "Winner"));
        assert!(matches!(&items[1], FilterItem::Winner(s) if s == "Alice"));
        assert!(matches!(&items[2], FilterItem::Winner(s) if s == "Bob"));
        assert!(matches!(&items[3], FilterItem::Header(s) if s == "Loser"));
        assert!(matches!(&items[4], FilterItem::Loser(s) if s == "Alice"));
        assert!(matches!(&items[5], FilterItem::Loser(s) if s == "Bob"));
        assert!(matches!(&items[6], FilterItem::Header(s) if s == "Participant"));
        assert!(matches!(&items[7], FilterItem::Participant(s) if s == "Alice"));
        assert!(matches!(&items[8], FilterItem::Participant(s) if s == "Bob"));
        assert!(matches!(&items[9], FilterItem::Header(s) if s == "Street"));
        assert!(matches!(
            &items[10],
            FilterItem::Street(RoundLabel::Preflop)
        ));
        assert!(matches!(&items[15], FilterItem::Header(s) if s == "Win Size"));
        assert!(matches!(
            &items[16],
            FilterItem::WinSize(ProfitBucket::Small)
        ));
        assert!(matches!(&items[20], FilterItem::Header(s) if s == "Loss Size"));
        assert!(matches!(
            &items[21],
            FilterItem::LossSize(ProfitBucket::Small)
        ));
        assert!(matches!(&items[25], FilterItem::Header(s) if s == "Players"));
        assert!(matches!(&items[26], FilterItem::PlayerCount(2)));
        assert!(matches!(&items[27], FilterItem::PlayerCount(6)));
        assert!(matches!(&items[28], FilterItem::ClearAll));
    }

    #[test]
    fn test_build_filter_items_empty_agents() {
        let items = build_filter_items(&[], &[]);
        // 6 headers (Winner, Loser, Participant, Street, Win Size, Loss Size)
        // + 0+0+0+5+4+4 items + ClearAll = 20
        assert_eq!(items.len(), 20);
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
