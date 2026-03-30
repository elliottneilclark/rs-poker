use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
};

use crate::tui::{
    state::{AgentDisplayData, GameLogEntry, Panel, TuiState},
    widgets::{
        filter_panel::render_filter_panel, game_log::render_game_log,
        profit_chart::render_profit_chart, progress_bar::render_progress,
        stats_table::render_stats_table, street_bars::render_street_bars,
    },
};

/// Layout rects for each interactive panel, used for mouse hit-testing.
pub struct PanelRects {
    pub table: Rect,
    pub game_log: Rect,
    pub filter: Rect,
}

impl PanelRects {
    /// Return the panel that contains the given (column, row) position, if any.
    pub fn hit_test(&self, col: u16, row: u16) -> Option<Panel> {
        if self.table.contains((col, row).into()) {
            Some(Panel::Table)
        } else if self.game_log.contains((col, row).into()) {
            Some(Panel::GameLog)
        } else if self.filter.contains((col, row).into()) {
            Some(Panel::Filter)
        } else {
            None
        }
    }

    /// Return the Rect for the given panel.
    pub fn rect_for(&self, panel: Panel) -> Rect {
        match panel {
            Panel::Table => self.table,
            Panel::GameLog => self.game_log,
            Panel::Filter => self.filter,
        }
    }
}

/// Render the overview screen. Returns the panel rects for mouse hit-testing
/// and animation effects.
pub fn render_overview(
    frame: &mut Frame,
    state: &TuiState,
    agents: &[AgentDisplayData],
    agent_names: &[String],
    game_log_entries: &[GameLogEntry],
    log_selected: Option<usize>,
) -> PanelRects {
    let main_chunks = Layout::vertical([
        Constraint::Percentage(50),
        Constraint::Percentage(48),
        Constraint::Length(1),
    ])
    .split(frame.area());

    // Top half: left column (stats table + street bars) | right column (profit chart)
    let top_columns = Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(main_chunks[0]);

    // Left column: stats table on top, street bars below
    let left_chunks = Layout::vertical([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(top_columns[0]);

    let table_focused = state.active_panel == Panel::Table;
    let log_focused = state.active_panel == Panel::GameLog;
    let filter_focused = state.active_panel == Panel::Filter;

    // Filter stats table rows when participant filter is active
    let filtered_agent_data: Vec<AgentDisplayData> = if state.filter.participants.is_empty() {
        agents.to_vec()
    } else {
        agents
            .iter()
            .filter(|a| state.filter.participants.contains(&a.name))
            .cloned()
            .collect()
    };

    render_stats_table(
        frame,
        left_chunks[0],
        &filtered_agent_data,
        state.table_selected,
        state.sort_col,
        table_focused,
    );
    render_street_bars(frame, left_chunks[1], &state.street_dist);

    // Right column: profit chart (full height)
    render_profit_chart(frame, top_columns[1], agents, state.profit_histories());

    // Bottom half: game_log (70%) | filter_panel (30%)
    let bottom_chunks =
        Layout::horizontal([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(main_chunks[1]);

    render_game_log(
        frame,
        bottom_chunks[0],
        game_log_entries,
        0,
        log_selected,
        log_focused,
    );

    let player_counts: Vec<usize> = state.distinct_player_counts.iter().copied().collect();
    render_filter_panel(
        frame,
        bottom_chunks[1],
        &state.filter,
        agent_names,
        &player_counts,
        filter_focused,
    );

    // Status bar
    render_progress(frame, main_chunks[2], state);

    PanelRects {
        table: left_chunks[0],
        game_log: bottom_chunks[0],
        filter: bottom_chunks[1],
    }
}
