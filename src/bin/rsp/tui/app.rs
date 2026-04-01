use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use rs_poker::open_hand_history::HandHistory;
use tachyonfx::Effect;
use tracing::warn;

use crate::tui::{
    effects,
    event::{Event, EventHandler, SimMessage},
    filtered_log::FilteredGameLog,
    hand_store::HandStore,
    screens::{game_detail, overview},
    state::{AgentDisplayData, GameLogEntry, GameResult, Panel, TuiState},
    terminal::{self, Tui},
};

/// Which screen the TUI is currently showing.
#[derive(Debug, Clone)]
pub enum Screen {
    Overview,
    /// Game detail screen, with the HandHistory and scroll offset.
    GameDetail {
        hand: Box<HandHistory>,
        scroll: u16,
    },
}

/// Main application state for the TUI.
pub struct App {
    pub state: TuiState,
    pub screen: Screen,
    pub should_quit: bool,
    /// Disk-backed store for on-demand HandHistory loading.
    pub hand_store: HandStore,
    /// Virtual game log backed by HandStore with filter support.
    pub filtered_log: FilteredGameLog,
    /// Active focus transition effect and the time it was created.
    focus_effect: Option<Effect>,
    /// Tracks time between frames for effect processing.
    last_frame: Instant,
    /// Cached panel rects from the last render, used for mouse hit-testing.
    panel_rects: Option<overview::PanelRects>,
}

impl App {
    pub fn new(games_target: Option<usize>) -> Self {
        Self {
            state: TuiState::new(games_target),
            screen: Screen::Overview,
            should_quit: false,
            hand_store: HandStore::none(),
            filtered_log: FilteredGameLog::new(),
            focus_effect: None,
            last_frame: Instant::now(),
            panel_rects: None,
        }
    }

    pub fn new_with_state(state: TuiState, hand_store: HandStore) -> Self {
        let mut filtered_log = FilteredGameLog::new();
        filtered_log.set_total(hand_store.len());
        Self {
            state,
            screen: Screen::Overview,
            should_quit: false,
            hand_store,
            filtered_log,
            focus_effect: None,
            last_frame: Instant::now(),
            panel_rects: None,
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        // Ctrl-C always quits, regardless of screen
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.should_quit = true;
            return;
        }
        match self.screen {
            Screen::Overview => self.handle_overview_key(key),
            Screen::GameDetail { .. } => self.handle_detail_key(key),
        }
    }

    /// Lines to scroll per mouse wheel tick.
    const SCROLL_LINES: isize = 3;

    pub fn handle_mouse(&mut self, mouse: MouseEvent) {
        match self.screen {
            Screen::Overview => self.handle_overview_mouse(mouse),
            Screen::GameDetail { .. } => self.handle_detail_mouse(mouse),
        }
    }

    fn handle_overview_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                if let Some(ref rects) = self.panel_rects
                    && let Some(panel) = rects.hit_test(mouse.column, mouse.row)
                    && panel != self.state.active_panel
                {
                    self.state.active_panel = panel;
                    self.focus_effect = Some(effects::border_chase());
                }
            }
            MouseEventKind::ScrollDown | MouseEventKind::ScrollUp => {
                let delta = if mouse.kind == MouseEventKind::ScrollDown {
                    Self::SCROLL_LINES
                } else {
                    -Self::SCROLL_LINES
                };
                if let Some(ref rects) = self.panel_rects
                    && let Some(panel) = rects.hit_test(mouse.column, mouse.row)
                {
                    let prev_panel = self.state.active_panel;
                    self.state.active_panel = panel;
                    self.move_panel_selection(delta);
                    self.state.active_panel = prev_panel;
                }
            }
            _ => {}
        }
    }

    fn handle_detail_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollDown => {
                if let Screen::GameDetail { ref mut scroll, .. } = self.screen {
                    *scroll = scroll.saturating_add(Self::SCROLL_LINES as u16);
                }
            }
            MouseEventKind::ScrollUp => {
                if let Screen::GameDetail { ref mut scroll, .. } = self.screen {
                    *scroll = scroll.saturating_sub(Self::SCROLL_LINES as u16);
                }
            }
            _ => {}
        }
    }

    /// Visible row count for the active panel, derived from cached layout rects.
    fn active_panel_page_size(&self) -> usize {
        self.panel_rects
            .as_ref()
            .map(|rects| {
                let rect = rects.rect_for(self.state.active_panel);
                rect.height.saturating_sub(4) as usize
            })
            .unwrap_or(10)
    }

    /// Maximum selectable index for the active panel.
    fn active_panel_max_index(&mut self) -> usize {
        match self.state.active_panel {
            Panel::Table => self.filtered_agents().len().saturating_sub(1),
            Panel::GameLog => self.filtered_log.total().saturating_sub(1),
            Panel::Filter => {
                let player_counts: Vec<usize> =
                    self.state.distinct_player_counts.iter().copied().collect();
                let items = crate::tui::widgets::filter_panel::build_filter_items(
                    &self.state.all_agent_names(),
                    &player_counts,
                );
                items.len().saturating_sub(1)
            }
        }
    }

    /// Return agent display data filtered by participant filter (matches rendered table).
    fn filtered_agents(&mut self) -> Vec<AgentDisplayData> {
        let agents = self.state.agent_display_data();
        if self.state.filter.participants.is_empty() {
            agents
        } else {
            agents
                .into_iter()
                .filter(|a| self.state.filter.participants.contains(&a.name))
                .collect()
        }
    }

    /// Move selection in the active panel by `delta` items (positive = down).
    fn move_panel_selection(&mut self, delta: isize) {
        let max = self.active_panel_max_index();
        let new_pos = |cur: usize| {
            if delta >= 0 {
                (cur + delta as usize).min(max)
            } else {
                cur.saturating_sub(delta.unsigned_abs())
            }
        };
        match self.state.active_panel {
            Panel::Table => {
                self.state.table_selected = Some(new_pos(self.state.table_selected.unwrap_or(0)));
            }
            Panel::GameLog => {
                self.state.log_selected = Some(new_pos(self.state.log_selected.unwrap_or(0)));
                self.keep_log_scroll_in_view();
            }
            Panel::Filter => {
                self.state.filter.selected = new_pos(self.state.filter.selected);
            }
        }
    }

    /// Set selection in the active panel to a specific index.
    fn move_panel_selection_to(&mut self, index: usize) {
        let max = self.active_panel_max_index();
        let clamped = index.min(max);
        match self.state.active_panel {
            Panel::Table => self.state.table_selected = Some(clamped),
            Panel::GameLog => {
                self.state.log_selected = Some(clamped);
                self.keep_log_scroll_in_view();
            }
            Panel::Filter => self.state.filter.selected = clamped,
        }
    }

    /// Adjust `log_scroll` so the selected game log entry stays in the visible window.
    fn keep_log_scroll_in_view(&mut self) {
        if let Some(selected) = self.state.log_selected {
            let page = self.active_panel_page_size();
            if page == 0 {
                return;
            }
            // Selection below visible window — scroll down
            if selected >= self.state.log_scroll + page {
                self.state.log_scroll = selected - page + 1;
            }
            // Selection above visible window — scroll up
            if selected < self.state.log_scroll {
                self.state.log_scroll = selected;
            }
        }
    }

    /// Center the game log scroll offset around the selected item.
    fn center_log_selection(&mut self) {
        if let Some(selected) = self.state.log_selected {
            let page = self.active_panel_page_size();
            self.state.log_scroll = selected.saturating_sub(page / 2);
        }
    }

    /// Approximate viewport height for the game detail scrollable area.
    fn detail_page_size(&self) -> u16 {
        crossterm::terminal::size()
            .map(|(_, h)| h.saturating_sub(10))
            .unwrap_or(20)
    }

    fn handle_overview_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.should_quit = true;
            }
            // Ctrl-modified vim motions
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let half = self.active_panel_page_size() / 2;
                self.move_panel_selection(half as isize);
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let half = self.active_panel_page_size() / 2;
                self.move_panel_selection(-(half as isize));
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let page = self.active_panel_page_size();
                self.move_panel_selection(page as isize);
            }
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let page = self.active_panel_page_size();
                self.move_panel_selection(-(page as isize));
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.move_panel_selection(1);
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.move_panel_selection(-1);
            }
            KeyCode::PageDown => {
                let page = self.active_panel_page_size();
                self.move_panel_selection(page as isize);
            }
            KeyCode::PageUp => {
                let page = self.active_panel_page_size();
                self.move_panel_selection(-(page as isize));
            }
            KeyCode::Char('g') | KeyCode::Home => {
                self.move_panel_selection_to(0);
            }
            KeyCode::Char('G') | KeyCode::End => {
                let max = self.active_panel_max_index();
                self.move_panel_selection_to(max);
            }
            KeyCode::Char('z') if self.state.active_panel == Panel::GameLog => {
                self.center_log_selection();
            }
            KeyCode::Tab => {
                self.state.active_panel = self.state.active_panel.next();
                self.focus_effect = Some(effects::border_chase());
            }
            KeyCode::BackTab => {
                self.state.active_panel = self.state.active_panel.prev();
                self.focus_effect = Some(effects::border_chase());
            }
            KeyCode::Char('s') => {
                self.state.sort_col = self.state.sort_col.next();
                self.state.invalidate_display_cache();
            }
            KeyCode::Enter if self.state.active_panel == Panel::GameLog => {
                if let Some(selected) = self.state.log_selected
                    && let Some(game_number) = self.filtered_log.game_number_at(selected)
                {
                    match self.hand_store.fetch(game_number) {
                        Ok(Some(hand)) => {
                            self.screen = Screen::GameDetail {
                                hand: Box::new(hand),
                                scroll: 0,
                            };
                        }
                        Ok(None) => {} // No OHH file or out of range
                        Err(e) => {
                            warn!("Failed to load hand {}: {}", game_number, e);
                        }
                    }
                }
            }
            KeyCode::Enter if self.state.active_panel == Panel::Table => {
                if let Some(selected) = self.state.table_selected {
                    let agents = self.filtered_agents();
                    if let Some(agent) = agents.get(selected) {
                        self.state.filter.toggle_participant(&agent.name);
                        self.clamp_table_selection();
                        self.reset_log_selection();
                    }
                }
            }
            KeyCode::Enter | KeyCode::Char(' ') if self.state.active_panel == Panel::Filter => {
                self.toggle_selected_filter();
            }
            KeyCode::Char('c') if self.state.active_panel == Panel::Filter => {
                self.state.filter.clear();
                self.reset_log_selection();
            }
            _ => {}
        }
    }

    fn toggle_selected_filter(&mut self) {
        let agent_names = self.state.all_agent_names();
        let player_counts: Vec<usize> = self.state.distinct_player_counts.iter().copied().collect();
        let items =
            crate::tui::widgets::filter_panel::build_filter_items(&agent_names, &player_counts);
        if let Some(item) = items.get(self.state.filter.selected) {
            use crate::tui::widgets::filter_panel::FilterItem;
            match item {
                FilterItem::Header(_) => {}
                FilterItem::Winner(name) => {
                    self.state.filter.toggle_winner(name);
                    self.reset_log_selection();
                }
                FilterItem::Loser(name) => {
                    self.state.filter.toggle_loser(name);
                    self.reset_log_selection();
                }
                FilterItem::Participant(name) => {
                    self.state.filter.toggle_participant(name);
                    self.reset_log_selection();
                }
                FilterItem::Street(round) => {
                    self.state.filter.toggle_street(*round);
                    self.reset_log_selection();
                }
                FilterItem::WinSize(bucket) => {
                    self.state.filter.toggle_win_size(*bucket);
                    self.reset_log_selection();
                }
                FilterItem::LossSize(bucket) => {
                    self.state.filter.toggle_loss_size(*bucket);
                    self.reset_log_selection();
                }
                FilterItem::PlayerCount(count) => {
                    self.state.filter.toggle_player_count(*count);
                    self.reset_log_selection();
                }
                FilterItem::ClearAll => {
                    self.state.filter.clear();
                    self.reset_log_selection();
                }
            }
        }
    }

    /// Clamp table selection to fit the current filtered agent list.
    fn clamp_table_selection(&mut self) {
        if let Some(sel) = self.state.table_selected {
            let max = self.filtered_agents().len().saturating_sub(1);
            if sel > max {
                self.state.table_selected = Some(max);
            }
        }
    }

    fn reset_log_selection(&mut self) {
        self.state.log_selected = None;
        self.state.log_scroll = 0;
        self.filtered_log
            .rebuild_filter(&self.state.filter, &self.hand_store);
    }

    fn handle_detail_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc | KeyCode::Backspace => {
                self.screen = Screen::Overview;
            }
            // Ctrl-modified vim motions
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let half = self.detail_page_size() / 2;
                if let Screen::GameDetail { ref mut scroll, .. } = self.screen {
                    *scroll = scroll.saturating_add(half);
                }
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let half = self.detail_page_size() / 2;
                if let Screen::GameDetail { ref mut scroll, .. } = self.screen {
                    *scroll = scroll.saturating_sub(half);
                }
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let page = self.detail_page_size();
                if let Screen::GameDetail { ref mut scroll, .. } = self.screen {
                    *scroll = scroll.saturating_add(page);
                }
            }
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let page = self.detail_page_size();
                if let Screen::GameDetail { ref mut scroll, .. } = self.screen {
                    *scroll = scroll.saturating_sub(page);
                }
            }
            KeyCode::Char('j') | KeyCode::Down => {
                if let Screen::GameDetail { ref mut scroll, .. } = self.screen {
                    *scroll = scroll.saturating_add(1);
                }
            }
            KeyCode::Char('k') | KeyCode::Up => {
                if let Screen::GameDetail { ref mut scroll, .. } = self.screen {
                    *scroll = scroll.saturating_sub(1);
                }
            }
            KeyCode::PageDown => {
                let page = self.detail_page_size();
                if let Screen::GameDetail { ref mut scroll, .. } = self.screen {
                    *scroll = scroll.saturating_add(page);
                }
            }
            KeyCode::PageUp => {
                let page = self.detail_page_size();
                if let Screen::GameDetail { ref mut scroll, .. } = self.screen {
                    *scroll = scroll.saturating_sub(page);
                }
            }
            KeyCode::Char('g') | KeyCode::Home => {
                if let Screen::GameDetail { ref mut scroll, .. } = self.screen {
                    *scroll = 0;
                }
            }
            KeyCode::Char('G') | KeyCode::End => {
                let page = self.detail_page_size();
                if let Screen::GameDetail {
                    ref mut scroll,
                    ref hand,
                } = self.screen
                {
                    let total = hand_content_lines(hand);
                    *scroll = total.saturating_sub(page);
                }
            }
            KeyCode::Char('Q') => {
                self.should_quit = true;
            }
            _ => {}
        }
    }

    pub fn handle_sim_message(&mut self, msg: SimMessage<GameResult>) {
        match msg {
            SimMessage::GameResult(result) => {
                let entry = GameLogEntry::new(
                    self.state.games_completed + 1,
                    result.agent_names.clone(),
                    result.profits.clone(),
                    result.ending_round,
                    result.big_blind,
                );
                self.state.update(result);
                self.filtered_log.on_new_game(&entry, &self.state.filter);
            }
            SimMessage::Completed => {
                self.state.completed = true;
            }
            SimMessage::Error(err) => {
                self.state.error = Some(err);
                self.state.completed = true;
            }
        }
    }
}

/// Approximate line count for the scrollable content in a game detail view.
fn hand_content_lines(hand: &HandHistory) -> u16 {
    game_detail::round_log_line_count(hand)
}

/// Main TUI event loop. Runs at ~30fps.
pub fn run_tui_loop(
    terminal: &mut Tui,
    app: &mut App,
    handler: &EventHandler<GameResult>,
) -> std::io::Result<()> {
    loop {
        // Compute frame delta for effect processing
        let dt = app.last_frame.elapsed();
        app.last_frame = Instant::now();

        // Take effect out so we can mutate it inside the draw closure
        let mut focus_effect = app.focus_effect.take();

        // Render current state (pre-compute before immutable borrow in draw closure)
        let agents = app.state.agent_display_data();
        let agent_names = app.state.all_agent_names();

        // Load game log entries for the visible window
        let (log_entries, log_selected) = if matches!(app.screen, Screen::Overview) {
            let page_size = app
                .panel_rects
                .as_ref()
                .map(|r| r.game_log.height.saturating_sub(2) as usize)
                .unwrap_or(20);
            app.filtered_log
                .ensure_window(app.state.log_scroll, page_size, &app.hand_store);
            let entries = app
                .filtered_log
                .window_entries(app.state.log_scroll, page_size)
                .to_vec();
            let selected = app
                .state
                .log_selected
                .map(|s| s.saturating_sub(app.state.log_scroll));
            (entries, selected)
        } else {
            (Vec::new(), None)
        };

        let mut new_panel_rects = None;
        terminal.draw(|frame| match &app.screen {
            Screen::Overview => {
                let rects = overview::render_overview(
                    frame,
                    &app.state,
                    &agents,
                    &agent_names,
                    &log_entries,
                    log_selected,
                );
                let focused_rect = rects.rect_for(app.state.active_panel);
                // Apply focus transition effect over the focused panel
                if let Some(ref mut effect) = focus_effect {
                    effect.process(dt, frame.buffer_mut(), focused_rect);
                }
                new_panel_rects = Some(rects);
            }
            Screen::GameDetail { hand, scroll } => {
                game_detail::render_detail(frame, hand, *scroll);
            }
        })?;

        app.panel_rects = new_panel_rects;

        // Put the effect back if it's still running
        if let Some(effect) = focus_effect
            && !effect.done()
        {
            app.focus_effect = Some(effect);
        }

        if app.should_quit {
            break;
        }

        // Handle events
        match handler.next()? {
            Event::Key(key) => {
                app.handle_key(key);
            }
            Event::Mouse(mouse) => {
                app.handle_mouse(mouse);
            }
            Event::Sim(msg) => {
                app.handle_sim_message(msg);
            }
            Event::Tick | Event::Resize => {}
        }

        // Drain all remaining sim messages to avoid falling behind
        while let Some(msg) = handler.try_recv_sim() {
            app.handle_sim_message(msg);
        }
    }
    Ok(())
}

/// Convenience function to setup terminal, run the TUI loop, and restore terminal.
pub fn run_app(app: &mut App, handler: &EventHandler<GameResult>) -> std::io::Result<()> {
    let mut terminal = terminal::setup_terminal()?;
    let result = run_tui_loop(&mut terminal, app, handler);
    terminal::restore_terminal(&mut terminal)?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::state::{GameResult, RoundLabel, SeatStats};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn test_q_sets_should_quit() {
        let mut app = App::new(Some(10));
        app.handle_key(key(KeyCode::Char('q')));
        assert!(app.should_quit);
    }

    #[test]
    fn test_esc_sets_should_quit() {
        let mut app = App::new(Some(10));
        app.handle_key(key(KeyCode::Esc));
        assert!(app.should_quit);
    }

    #[test]
    fn test_ctrl_c_quits_from_overview() {
        let mut app = App::new(Some(10));
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(app.should_quit);
    }

    #[test]
    fn test_ctrl_c_quits_from_detail() {
        let mut app = App::new(Some(10));
        app.screen = Screen::GameDetail {
            hand: Box::new(make_test_hand()),
            scroll: 0,
        };
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(app.should_quit);
    }

    #[test]
    fn test_tab_switches_panel() {
        let mut app = App::new(Some(10));
        assert_eq!(app.state.active_panel, Panel::Table);
        app.handle_key(key(KeyCode::Tab));
        assert_eq!(app.state.active_panel, Panel::GameLog);
        app.handle_key(key(KeyCode::Tab));
        assert_eq!(app.state.active_panel, Panel::Filter);
        app.handle_key(key(KeyCode::Tab));
        assert_eq!(app.state.active_panel, Panel::Table);
    }

    #[test]
    fn test_shift_tab_switches_panel_reverse() {
        let mut app = App::new(Some(10));
        assert_eq!(app.state.active_panel, Panel::Table);
        app.handle_key(key(KeyCode::BackTab));
        assert_eq!(app.state.active_panel, Panel::Filter);
        app.handle_key(key(KeyCode::BackTab));
        assert_eq!(app.state.active_panel, Panel::GameLog);
        app.handle_key(key(KeyCode::BackTab));
        assert_eq!(app.state.active_panel, Panel::Table);
    }

    #[test]
    fn test_s_cycles_sort_column() {
        let mut app = App::new(Some(10));
        use crate::tui::widgets::stats_table::SortColumn;
        assert_eq!(app.state.sort_col, SortColumn::Profit);
        app.handle_key(key(KeyCode::Char('s')));
        assert_eq!(app.state.sort_col, SortColumn::Games);
    }

    #[test]
    fn test_backspace_on_detail_returns_to_overview() {
        let mut app = App::new(Some(10));
        app.screen = Screen::GameDetail {
            hand: Box::new(make_test_hand()),
            scroll: 0,
        };
        app.handle_key(key(KeyCode::Backspace));
        assert!(matches!(app.screen, Screen::Overview));
    }

    #[test]
    fn test_sim_message_updates_state() {
        let mut app = App::new(Some(10));
        let result = GameResult {
            agent_names: vec!["A".into()],
            profits: vec![10.0],
            ending_round: RoundLabel::Preflop,
            seat_stats: vec![SeatStats::default()],
            big_blind: 10.0,
        };
        app.handle_sim_message(SimMessage::GameResult(result));
        assert_eq!(app.state.games_completed, 1);
    }

    #[test]
    fn test_completed_message_sets_flag() {
        let mut app = App::new(Some(10));
        app.handle_sim_message(SimMessage::<GameResult>::Completed);
        assert!(app.state.completed);
    }

    fn add_game(app: &mut App, names: &[&str], profits: &[f32], round: RoundLabel) {
        let seat_stats: Vec<SeatStats> = profits
            .iter()
            .map(|&p| SeatStats {
                total_profit: p,
                hands_played: 1,
                ..SeatStats::default()
            })
            .collect();
        app.handle_sim_message(SimMessage::GameResult(GameResult {
            agent_names: names.iter().map(|s| s.to_string()).collect(),
            profits: profits.to_vec(),
            ending_round: round,
            seat_stats,
            big_blind: 10.0,
        }));
    }

    #[test]
    fn test_enter_on_table_toggles_participant_filter() {
        let mut app = App::new(Some(10));
        add_game(
            &mut app,
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::River,
        );

        // Select first agent row and press Enter
        app.state.active_panel = Panel::Table;
        app.state.table_selected = Some(0);
        app.handle_key(key(KeyCode::Enter));

        let agents = app.state.agent_display_data();
        let top_agent = &agents[0].name;
        assert!(app.state.filter.participants.contains(top_agent));

        // Press Enter again to toggle off
        app.handle_key(key(KeyCode::Enter));
        assert!(!app.state.filter.participants.contains(top_agent));
    }

    #[test]
    fn test_filter_panel_navigation() {
        let mut app = App::new(Some(10));
        add_game(
            &mut app,
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::River,
        );

        app.state.active_panel = Panel::Filter;
        assert_eq!(app.state.filter.selected, 0);

        app.handle_key(key(KeyCode::Char('j')));
        assert_eq!(app.state.filter.selected, 1);

        app.handle_key(key(KeyCode::Char('k')));
        assert_eq!(app.state.filter.selected, 0);

        // Can't go below 0
        app.handle_key(key(KeyCode::Char('k')));
        assert_eq!(app.state.filter.selected, 0);
    }

    #[test]
    fn test_filter_panel_enter_toggles_winner() {
        let mut app = App::new(Some(10));
        add_game(
            &mut app,
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::River,
        );

        app.state.active_panel = Panel::Filter;
        // Item 0 is "Winner" header, item 1 is Winner("Alice")
        app.state.filter.selected = 1;
        app.handle_key(key(KeyCode::Enter));
        assert!(app.state.filter.winners.contains("Alice"));

        // Toggle off
        app.handle_key(key(KeyCode::Enter));
        assert!(!app.state.filter.winners.contains("Alice"));
    }

    #[test]
    fn test_filter_panel_space_toggles_item() {
        let mut app = App::new(Some(10));
        add_game(
            &mut app,
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::River,
        );

        app.state.active_panel = Panel::Filter;
        app.state.filter.selected = 1; // Winner("Alice")
        app.handle_key(key(KeyCode::Char(' ')));
        assert!(app.state.filter.winners.contains("Alice"));
    }

    #[test]
    fn test_filter_panel_enter_toggles_loser() {
        let mut app = App::new(Some(10));
        add_game(
            &mut app,
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::River,
        );

        app.state.active_panel = Panel::Filter;
        // Header "Winner" + 2 winners + Header "Loser" = index 3 is header
        // index 4 is Loser("Alice")
        app.state.filter.selected = 4;
        app.handle_key(key(KeyCode::Enter));
        assert!(app.state.filter.losers.contains("Alice"));

        app.handle_key(key(KeyCode::Enter));
        assert!(!app.state.filter.losers.contains("Alice"));
    }

    #[test]
    fn test_filter_panel_enter_toggles_win_size() {
        let mut app = App::new(Some(10));
        add_game(
            &mut app,
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::River,
        );

        app.state.active_panel = Panel::Filter;
        // Winner(2) + Loser(2) + Participant(2) + headers(3) = 9
        // + Header "Street" = 10, + 5 streets = 15
        // + Header "Win Size" = 16, first WinSize = 17
        // With 2 agents: indices are:
        // 0: Header Winner, 1-2: winners, 3: Header Loser, 4-5: losers,
        // 6: Header Participant, 7-8: participants,
        // 9: Header Street, 10-14: streets,
        // 15: Header Win Size, 16: WinSize(Small)
        app.state.filter.selected = 16;
        app.handle_key(key(KeyCode::Enter));
        assert!(
            app.state
                .filter
                .win_sizes
                .contains(&crate::tui::state::ProfitBucket::Small)
        );
    }

    #[test]
    fn test_filter_panel_enter_toggles_loss_size() {
        let mut app = App::new(Some(10));
        add_game(
            &mut app,
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::River,
        );

        app.state.active_panel = Panel::Filter;
        // After Win Size section: 16-19 are WinSize buckets
        // 20: Header Loss Size, 21: LossSize(Small)
        app.state.filter.selected = 21;
        app.handle_key(key(KeyCode::Enter));
        assert!(
            app.state
                .filter
                .loss_sizes
                .contains(&crate::tui::state::ProfitBucket::Small)
        );
    }

    #[test]
    fn test_filter_panel_header_is_noop() {
        let mut app = App::new(Some(10));
        add_game(
            &mut app,
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::River,
        );

        app.state.active_panel = Panel::Filter;
        app.state.filter.selected = 0; // Header("Winner")
        app.handle_key(key(KeyCode::Enter));
        assert!(!app.state.filter.is_active());
    }

    #[test]
    fn test_c_clears_filters() {
        let mut app = App::new(Some(10));
        add_game(
            &mut app,
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::River,
        );

        app.state.active_panel = Panel::Filter;
        app.state.filter.toggle_winner("Alice");
        app.state.filter.toggle_street(RoundLabel::River);
        assert!(app.state.filter.is_active());

        app.handle_key(key(KeyCode::Char('c')));
        assert!(!app.state.filter.is_active());
    }

    #[test]
    fn test_filter_resets_log_selection() {
        let mut app = App::new(Some(10));
        add_game(
            &mut app,
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::River,
        );
        add_game(&mut app, &["Alice", "Bob"], &[-5.0, 5.0], RoundLabel::Flop);

        app.state.log_selected = Some(1);
        app.state.log_scroll = 5;

        // Toggling a filter via Table Enter should reset log selection
        app.state.active_panel = Panel::Table;
        app.state.table_selected = Some(0);
        app.handle_key(key(KeyCode::Enter));

        assert_eq!(app.state.log_selected, None);
        assert_eq!(app.state.log_scroll, 0);
    }

    #[test]
    fn test_game_log_j_uses_filtered_length() {
        let mut app = App::new(Some(10));
        add_game(
            &mut app,
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::River,
        );
        add_game(&mut app, &["Alice", "Bob"], &[-5.0, 5.0], RoundLabel::Flop);
        add_game(&mut app, &["Alice", "Bob"], &[3.0, -3.0], RoundLabel::River);

        // Simulate River-only filter: games 1 and 3 pass (2 of 3)
        app.state.filter.toggle_street(RoundLabel::River);
        app.filtered_log = FilteredGameLog::test_with_filter(3, vec![1, 3]);
        app.state.active_panel = Panel::GameLog;

        app.handle_key(key(KeyCode::Char('j'))); // 0
        app.handle_key(key(KeyCode::Char('j'))); // 1
        app.handle_key(key(KeyCode::Char('j'))); // still 1 (max)
        assert_eq!(app.state.log_selected, Some(1));
    }

    #[test]
    fn test_game_log_scroll_follows_selection() {
        let mut app = App::new(Some(200));
        // Add 100 games
        for _ in 0..100 {
            add_game(&mut app, &["A", "B"], &[10.0, -10.0], RoundLabel::River);
        }

        app.state.active_panel = Panel::GameLog;

        // panel_rects is None in tests, so active_panel_page_size() returns 10.
        // Navigate down 50 times (well past the visible window of ~10).
        for _ in 0..50 {
            app.handle_key(key(KeyCode::Char('j')));
        }

        assert_eq!(app.state.log_selected, Some(50));
        // log_scroll must have advanced so the selected item is visible.
        // With page_size=10, scroll should be at least 50 - 10 + 1 = 41.
        assert!(
            app.state.log_scroll >= 41,
            "log_scroll ({}) should be >= 41 to keep selection 50 visible in a 10-row page",
            app.state.log_scroll
        );

        // Now scroll back up
        for _ in 0..50 {
            app.handle_key(key(KeyCode::Char('k')));
        }

        assert_eq!(app.state.log_selected, Some(0));
        assert_eq!(
            app.state.log_scroll, 0,
            "log_scroll should return to 0 when selection is at top"
        );
    }

    fn mouse_scroll(kind: MouseEventKind, col: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind,
            column: col,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }

    /// Set up fake panel rects so mouse hit-testing works in tests.
    fn set_test_panel_rects(app: &mut App) {
        use crate::tui::screens::overview::PanelRects;
        use ratatui::layout::Rect;
        app.panel_rects = Some(PanelRects {
            table: Rect::new(0, 0, 80, 20),
            game_log: Rect::new(0, 20, 60, 20),
            filter: Rect::new(60, 20, 20, 20),
        });
    }

    #[test]
    fn test_scroll_wheel_on_game_log() {
        let mut app = App::new(Some(200));
        for _ in 0..50 {
            add_game(&mut app, &["A", "B"], &[10.0, -10.0], RoundLabel::River);
        }
        set_test_panel_rects(&mut app);
        // game_log rect is 20 rows, page_size = 20 - 4 = 16

        // Scroll down over the game log area (col=10, row=25 is inside game_log rect)
        // 7 scrolls * 3 lines = selection 21, which overflows page_size of 16
        for _ in 0..7 {
            app.handle_mouse(mouse_scroll(MouseEventKind::ScrollDown, 10, 25));
        }
        assert_eq!(app.state.log_selected, Some(21));
        assert!(
            app.state.log_scroll > 0,
            "log_scroll ({}) should be > 0 when selection exceeds page size",
            app.state.log_scroll
        );

        // Scroll back up past the top
        for _ in 0..10 {
            app.handle_mouse(mouse_scroll(MouseEventKind::ScrollUp, 10, 25));
        }
        assert_eq!(app.state.log_selected, Some(0));
        assert_eq!(app.state.log_scroll, 0);
    }

    #[test]
    fn test_scroll_wheel_on_table() {
        let mut app = App::new(Some(10));
        // Need multiple agents to scroll through
        add_game(
            &mut app,
            &["A", "B", "C", "D", "E"],
            &[10.0, -5.0, -3.0, -1.0, -1.0],
            RoundLabel::River,
        );
        set_test_panel_rects(&mut app);

        // Scroll down over table area (col=10, row=5)
        app.handle_mouse(mouse_scroll(MouseEventKind::ScrollDown, 10, 5));
        assert_eq!(app.state.table_selected, Some(3));

        app.handle_mouse(mouse_scroll(MouseEventKind::ScrollUp, 10, 5));
        assert_eq!(app.state.table_selected, Some(0));
    }

    #[test]
    fn test_scroll_wheel_on_detail_screen() {
        let mut app = App::new(Some(10));
        app.screen = Screen::GameDetail {
            hand: Box::new(make_test_hand()),
            scroll: 0,
        };

        app.handle_mouse(mouse_scroll(MouseEventKind::ScrollDown, 10, 10));
        if let Screen::GameDetail { scroll, .. } = app.screen {
            assert_eq!(scroll, 3);
        } else {
            panic!("expected GameDetail screen");
        }

        app.handle_mouse(mouse_scroll(MouseEventKind::ScrollUp, 10, 10));
        if let Screen::GameDetail { scroll, .. } = app.screen {
            assert_eq!(scroll, 0);
        } else {
            panic!("expected GameDetail screen");
        }
    }

    fn make_test_hand() -> HandHistory {
        use rs_poker::open_hand_history::*;
        HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "test".into(),
            network_name: "test".into(),
            internal_version: "1.0".into(),
            tournament: false,
            tournament_info: None,
            game_number: "1".into(),
            start_date_utc: None,
            table_name: "test".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: None,
            table_size: 2,
            currency: "USD".into(),
            dealer_seat: 0,
            small_blind_amount: 5.0,
            big_blind_amount: 10.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players: vec![],
            rounds: vec![],
            pots: vec![],
            tournament_bounties: None,
        }
    }
}
