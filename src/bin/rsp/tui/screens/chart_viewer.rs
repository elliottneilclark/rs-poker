//! Preflop chart viewer screen.
//!
//! Renders a header of seat tabs, a 13x13 color-only hand grid, range-weighted
//! action totals for the selected seat/scenario, and hover detail for the cell
//! under the cursor.

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Gauge, Paragraph},
};

use rs_poker::arena::cfr::PreflopChartConfig;
use rs_poker::holdem::{PreflopActionType, PreflopScenario, PreflopStrategy};

use crate::tui::{
    theme,
    widgets::hand_grid::{
        action_color, action_label, blended_color, hand_at, render_hand_grid, seat_totals,
    },
};

/// State for the preflop chart viewer.
pub struct ChartViewerState {
    pub config: PreflopChartConfig,
    pub config_name: String,
    /// Number of seats to display tabs for. Usually 6 for a 6-max config.
    pub num_seats: usize,
    /// Currently selected seat index (BB-relative: 0=BB, 1=SB, 2=BTN, ...).
    pub current_seat: usize,
    /// Currently selected decision scenario.
    pub current_scenario: PreflopScenario,
    /// Grid hover cursor: `(row, col)` where 0 ≤ row,col < 13.
    pub hover: (usize, usize),
    /// When true, the config did not ship a preflop chart and the grid is
    /// synthesized. The header shows a warning.
    pub synthesized: bool,
    /// Banner shown in place of the grid (for configs with no preflop info).
    pub banner: Option<String>,
}

impl ChartViewerState {
    pub fn new(
        config: PreflopChartConfig,
        config_name: String,
        num_seats: usize,
        synthesized: bool,
    ) -> Self {
        let mut state = Self {
            config,
            config_name,
            num_seats: num_seats.max(1),
            current_seat: 0,
            current_scenario: PreflopScenario::Rfi,
            hover: (0, 0),
            synthesized,
            banner: None,
        };
        state.ensure_non_empty_scenario();
        state
    }

    pub fn set_scenario(&mut self, scenario: PreflopScenario) {
        self.current_scenario = scenario;
    }

    pub fn next_scenario(&mut self) {
        let all = PreflopScenario::all();
        let idx = all
            .iter()
            .position(|s| *s == self.current_scenario)
            .unwrap_or(0);
        self.current_scenario = all[(idx + 1) % all.len()];
    }

    pub fn prev_scenario(&mut self) {
        let all = PreflopScenario::all();
        let idx = all
            .iter()
            .position(|s| *s == self.current_scenario)
            .unwrap_or(0);
        self.current_scenario = all[(idx + all.len() - 1) % all.len()];
    }

    pub fn with_banner(mut self, banner: impl Into<String>) -> Self {
        self.banner = Some(banner.into());
        self
    }

    pub fn move_hover(&mut self, d_row: isize, d_col: isize) {
        let (r, c) = self.hover;
        let r = (r as isize + d_row).clamp(0, 12) as usize;
        let c = (c as isize + d_col).clamp(0, 12) as usize;
        self.hover = (r, c);
    }

    pub fn set_seat(&mut self, seat: usize) {
        if self.num_seats == 0 {
            self.current_seat = 0;
        } else {
            self.current_seat = seat % self.num_seats;
        }
        self.ensure_non_empty_scenario();
    }

    /// Returns true if the chart at the current (seat, scenario) has any
    /// entries.
    fn current_scenario_has_data(&self) -> bool {
        !self
            .config
            .chart_for(self.current_seat, self.current_scenario)
            .is_empty()
    }

    /// If the current scenario has no data for the current seat, advance to
    /// the first scenario that does. No-op if every scenario is empty (truly
    /// blank position) or if the current one already has data.
    ///
    /// This makes seat switching "do the sensible thing" — e.g., picking BB
    /// lands on `vs Open` (the only scenario with data for BB) instead of
    /// RFI (structurally empty, since BB doesn't open).
    fn ensure_non_empty_scenario(&mut self) {
        if self.current_scenario_has_data() {
            return;
        }
        for scenario in PreflopScenario::all() {
            if !self
                .config
                .chart_for(self.current_seat, scenario)
                .is_empty()
            {
                self.current_scenario = scenario;
                return;
            }
        }
    }

    pub fn next_seat(&mut self) {
        self.set_seat(self.current_seat + 1);
    }

    pub fn prev_seat(&mut self) {
        if self.current_seat == 0 {
            self.set_seat(self.num_seats.saturating_sub(1));
        } else {
            self.set_seat(self.current_seat - 1);
        }
    }
}

/// Canonical label for a BB-relative seat index, matching
/// `PreflopChartConfig::calculate_position` semantics.
///
/// 0=BB, 1=SB, 2=BTN, 3=CO, 4=HJ, 5=UTG, then UTG+1 / UTG+2 / ... for larger
/// tables.
pub fn seat_label(seat: usize, num_seats: usize) -> String {
    match (seat, num_seats) {
        (0, _) => "BB".to_string(),
        (1, 2) => "BTN".to_string(),
        (1, _) => "SB".to_string(),
        (2, _) => "BTN".to_string(),
        (3, _) => "CO".to_string(),
        (4, _) => "HJ".to_string(),
        (5, _) => "UTG".to_string(),
        (n, _) => format!("UTG+{}", n - 5),
    }
}

/// Layout rects produced by `render_chart_viewer`, exposed for mouse
/// hit-testing by the event loop.
pub struct ChartViewerRects {
    pub seat_tabs: Vec<Rect>,
    pub scenario_tabs: Vec<(PreflopScenario, Rect)>,
    pub grid: Rect,
}

pub fn render_chart_viewer(frame: &mut Frame, state: &ChartViewerState) -> ChartViewerRects {
    let area = frame.area();
    let outer = theme::chart_block(&format!(
        "Preflop Charts: {}{}",
        state.config_name,
        if state.synthesized {
            " (synthesized)"
        } else {
            ""
        }
    ));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    let chunks = Layout::vertical([
        Constraint::Length(1), // Seat tab row
        Constraint::Length(1), // Scenario tab row
        Constraint::Length(1), // Config params row
        Constraint::Min(14),   // Grid + right panel
        Constraint::Length(1), // Legend
        Constraint::Length(1), // Keybindings
    ])
    .split(inner);

    let seat_tabs = render_seat_tabs(frame, chunks[0], state);
    let scenario_tabs = render_scenario_tabs(frame, chunks[1], state);
    render_config_params(frame, chunks[2], state);

    let body = Layout::horizontal([Constraint::Min(41), Constraint::Length(44)]).split(chunks[3]);

    let grid_rect = body[0];
    if let Some(msg) = &state.banner {
        let banner = Paragraph::new(msg.as_str()).style(Style::default().fg(theme::YELLOW));
        frame.render_widget(banner, grid_rect);
    } else {
        let chart = state
            .config
            .chart_for(state.current_seat, state.current_scenario);
        render_hand_grid(frame, grid_rect, chart, Some(state.hover));
    }

    render_right_panel(frame, body[1], state);
    render_legend(frame, chunks[4]);
    render_keybindings(frame, chunks[5]);

    ChartViewerRects {
        seat_tabs,
        scenario_tabs,
        grid: grid_rect,
    }
}

fn render_scenario_tabs(
    frame: &mut Frame,
    area: Rect,
    state: &ChartViewerState,
) -> Vec<(PreflopScenario, Rect)> {
    let mut rects: Vec<(PreflopScenario, Rect)> = Vec::with_capacity(4);
    let mut spans = vec![Span::styled(
        "Scenario: ",
        Style::default().fg(theme::SUBTEXT1),
    )];
    let mut x = area.x + "Scenario: ".len() as u16;
    for scenario in PreflopScenario::all() {
        let is_current = scenario == state.current_scenario;
        let is_empty = state
            .config
            .chart_for(state.current_seat, scenario)
            .is_empty();
        let tab_text = if is_current {
            format!("▶{}◀ ", scenario.label())
        } else {
            format!(" {}  ", scenario.label())
        };
        let style = if is_current {
            Style::default()
                .fg(theme::FOCUS_COLOR)
                .add_modifier(Modifier::BOLD)
        } else if is_empty {
            // Empty chart for this (seat, scenario) — dim so users know there's
            // nothing to see without having to click through.
            Style::default().fg(theme::OVERLAY0)
        } else {
            Style::default().fg(theme::SUBTEXT0)
        };
        let width = tab_text.chars().count() as u16;
        let rect = Rect::new(
            x,
            area.y,
            width.min(area.width.saturating_sub(x - area.x)),
            1,
        );
        spans.push(Span::styled(tab_text, style));
        rects.push((scenario, rect));
        x += width;
        if x >= area.x + area.width {
            break;
        }
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
    rects
}

fn render_seat_tabs(frame: &mut Frame, area: Rect, state: &ChartViewerState) -> Vec<Rect> {
    let mut rects = Vec::with_capacity(state.num_seats);
    let mut spans = vec![Span::styled(
        "Seats: ",
        Style::default().fg(theme::SUBTEXT1),
    )];
    let mut x = area.x + "Seats: ".len() as u16;
    for seat in 0..state.num_seats {
        let label = seat_label(seat, state.num_seats);
        let is_current = seat == state.current_seat;
        let tab_text = if is_current {
            format!("▶{}◀ ", label)
        } else {
            format!(" {}  ", label)
        };
        let style = if is_current {
            Style::default()
                .fg(theme::FOCUS_COLOR)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme::SUBTEXT0)
        };
        let width = tab_text.chars().count() as u16;
        let rect = Rect::new(
            x,
            area.y,
            width.min(area.width.saturating_sub(x - area.x)),
            1,
        );
        spans.push(Span::styled(tab_text, style));
        rects.push(rect);
        x += width;
        if x >= area.x + area.width {
            break;
        }
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
    rects
}

fn render_config_params(frame: &mut Frame, area: Rect, state: &ChartViewerState) {
    let text = if state.synthesized {
        format!(
            "  (no preflop_config — synthesized from {})",
            state.config_name
        )
    } else {
        format!(
            "  raise {:.2}bb   3bet×{:.1}   4bet+×{:.1}   positions: {}",
            state.config.raise_size_bb,
            state.config.three_bet_multiplier,
            state.config.four_bet_plus_multiplier,
            state.config.positions.len(),
        )
    };
    let para = Paragraph::new(text).style(Style::default().fg(theme::SUBTEXT0));
    frame.render_widget(para, area);
}

fn render_right_panel(frame: &mut Frame, area: Rect, state: &ChartViewerState) {
    let sub = Layout::vertical([Constraint::Length(5), Constraint::Min(5)]).split(area);

    if state.banner.is_none() {
        let chart = state
            .config
            .chart_for(state.current_seat, state.current_scenario);
        let totals = seat_totals(chart);
        render_totals_block(
            frame,
            sub[0],
            &format!(
                "{} · {} — range totals",
                seat_label(state.current_seat, state.num_seats),
                state.current_scenario.label(),
            ),
            &totals,
        );

        let hover_hand = hand_at(state.hover.0, state.hover.1);
        let hover_strategy = chart.get_or_fold(&hover_hand);
        let hover_rows = strategy_rows(&hover_strategy);
        render_totals_block(
            frame,
            sub[1],
            &format!("Hover: {} · {}", hover_hand, state.current_scenario.label()),
            &hover_rows,
        );
    } else {
        let para = Paragraph::new("No seat data").style(Style::default().fg(theme::SUBTEXT0));
        frame.render_widget(para, area);
    }
}

fn strategy_rows(strategy: &PreflopStrategy) -> [(PreflopActionType, f32); 3] {
    [
        (PreflopActionType::Fold, strategy.fold_freq()),
        (PreflopActionType::Call, strategy.call()),
        (PreflopActionType::Raise, strategy.raise()),
    ]
}

fn render_totals_block(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    rows: &[(PreflopActionType, f32)],
) {
    let block = theme::chart_block(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    render_action_bars(frame, inner, rows);
}

fn render_action_bars(frame: &mut Frame, area: Rect, rows: &[(PreflopActionType, f32)]) {
    if area.height == 0 {
        return;
    }
    // Canonical display order so the panel doesn't jitter as the hover moves.
    let canonical = [
        PreflopActionType::Raise,
        PreflopActionType::Call,
        PreflopActionType::Fold,
    ];

    let lookup = |action: PreflopActionType| {
        rows.iter()
            .find(|(a, _)| *a == action)
            .map(|(_, f)| *f)
            .unwrap_or(0.0)
    };

    let line_count = canonical.len().min(area.height as usize);
    let constraints: Vec<Constraint> = (0..line_count).map(|_| Constraint::Length(1)).collect();
    let chunks = Layout::vertical(constraints).split(Rect::new(
        area.x,
        area.y,
        area.width,
        line_count as u16,
    ));

    for (i, action) in canonical.iter().take(line_count).enumerate() {
        let freq = lookup(*action);
        let label = format!("{:<5}", action_label(*action));
        let pct = format!("{:>4}%", (freq * 100.0).round() as i32);

        let row = chunks[i];
        if row.width < 14 {
            let text = format!("{} {}", label, pct);
            frame.render_widget(
                Paragraph::new(text).style(Style::default().fg(action_color(*action))),
                row,
            );
            continue;
        }
        let parts = Layout::horizontal([
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Min(4),
        ])
        .split(row);
        frame.render_widget(
            Paragraph::new(label).style(Style::default().fg(action_color(*action))),
            parts[0],
        );
        frame.render_widget(
            Paragraph::new(pct).style(Style::default().fg(theme::SUBTEXT1)),
            parts[1],
        );
        let gauge = Gauge::default()
            .ratio(freq.clamp(0.0, 1.0) as f64)
            .label("")
            .gauge_style(
                Style::default()
                    .fg(action_color(*action))
                    .bg(theme::SURFACE1),
            );
        frame.render_widget(gauge, parts[2]);
    }
}

fn render_legend(frame: &mut Frame, area: Rect) {
    let swatches = [
        PreflopActionType::Raise,
        PreflopActionType::Call,
        PreflopActionType::Fold,
    ];
    let pure = |action: PreflopActionType| -> PreflopStrategy {
        match action {
            PreflopActionType::Raise => PreflopStrategy::pure_raise(),
            PreflopActionType::Call => PreflopStrategy::pure_call(),
            PreflopActionType::Fold => PreflopStrategy::fold(),
        }
    };
    let mut spans: Vec<Span> = Vec::with_capacity(swatches.len() * 2 + 1);
    for action in swatches {
        let strategy = pure(action);
        let bg = blended_color(Some(&strategy));
        spans.push(Span::styled("   ", Style::default().bg(bg).fg(theme::TEXT)));
        spans.push(Span::styled(
            format!(" {}  ", action_label(action)),
            Style::default().fg(theme::SUBTEXT1),
        ));
    }
    spans.push(Span::styled(
        "(blended = mixed strategy)",
        Style::default().fg(theme::OVERLAY0),
    ));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_keybindings(frame: &mut Frame, area: Rect) {
    let spans = vec![
        Span::styled("[1-9]", theme::keybinding_key_style()),
        Span::styled(" seat  ", Style::default().fg(theme::OVERLAY0)),
        Span::styled("[r/o/t/f]", theme::keybinding_key_style()),
        Span::styled(" scenario  ", Style::default().fg(theme::OVERLAY0)),
        Span::styled("[s]", theme::keybinding_key_style()),
        Span::styled(" cycle  ", Style::default().fg(theme::OVERLAY0)),
        Span::styled("[hjkl]", theme::keybinding_key_style()),
        Span::styled(" move  ", Style::default().fg(theme::OVERLAY0)),
        Span::styled("[Tab]", theme::keybinding_key_style()),
        Span::styled(" next seat  ", Style::default().fg(theme::OVERLAY0)),
        Span::styled("[q]", theme::keybinding_key_style()),
        Span::styled(" quit", Style::default().fg(theme::OVERLAY0)),
    ];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    use rs_poker::arena::cfr::PositionCharts;
    use rs_poker::core::Value;
    use rs_poker::holdem::PreflopHand;

    fn simple_position_charts() -> PositionCharts {
        let mut charts = PositionCharts::default();
        charts.rfi.set(
            PreflopHand::new(Value::Ace, Value::Ace, false),
            PreflopStrategy::pure_raise(),
        );
        charts
    }

    fn simple_state(num_seats: usize) -> ChartViewerState {
        let config = PreflopChartConfig::with_single_position(simple_position_charts());
        ChartViewerState::new(config, "test".to_string(), num_seats, false)
    }

    #[test]
    fn seat_labels_6max() {
        assert_eq!(seat_label(0, 6), "BB");
        assert_eq!(seat_label(1, 6), "SB");
        assert_eq!(seat_label(2, 6), "BTN");
        assert_eq!(seat_label(3, 6), "CO");
        assert_eq!(seat_label(4, 6), "HJ");
        assert_eq!(seat_label(5, 6), "UTG");
    }

    #[test]
    fn seat_labels_heads_up_uses_btn_for_one() {
        assert_eq!(seat_label(0, 2), "BB");
        assert_eq!(seat_label(1, 2), "BTN");
    }

    #[test]
    fn seat_labels_large_table_appends_plus() {
        assert_eq!(seat_label(6, 9), "UTG+1");
        assert_eq!(seat_label(8, 9), "UTG+3");
    }

    #[test]
    fn move_hover_clamps() {
        let mut state = simple_state(6);
        state.hover = (0, 0);
        state.move_hover(-1, -1);
        assert_eq!(state.hover, (0, 0));
        state.move_hover(100, 100);
        assert_eq!(state.hover, (12, 12));
    }

    #[test]
    fn seat_navigation_wraps() {
        let mut state = simple_state(6);
        state.set_seat(5);
        state.next_seat();
        assert_eq!(state.current_seat, 0);
        state.prev_seat();
        assert_eq!(state.current_seat, 5);
    }

    #[test]
    fn render_does_not_panic() {
        let state = simple_state(6);
        let backend = TestBackend::new(100, 25);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                let _ = render_chart_viewer(frame, &state);
            })
            .unwrap();
    }

    #[test]
    fn render_with_banner_does_not_panic() {
        let state = simple_state(6).with_banner("no chart");
        let backend = TestBackend::new(100, 25);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                let _ = render_chart_viewer(frame, &state);
            })
            .unwrap();
    }

    /// Regression: switching to a seat whose current scenario has no data
    /// should auto-advance to the first scenario that does. The motivating
    /// case is BB, whose RFI chart is structurally empty in most configs —
    /// previously the user saw an all-fold grid with no hint that vs_open
    /// had data.
    #[test]
    fn set_seat_auto_switches_to_non_empty_scenario() {
        use rs_poker::arena::cfr::PositionCharts;

        // Position 0 (BB) has only vs_open data; position 1 has only rfi.
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        let mut bb = PositionCharts::default();
        bb.vs_open.set(aa, PreflopStrategy::pure_call());
        let mut sb = PositionCharts::default();
        sb.rfi.set(aa, PreflopStrategy::pure_raise());

        let config = PreflopChartConfig::new(vec![bb, sb]);
        let mut state = ChartViewerState::new(config, "test".into(), 6, false);

        // Start on seat 1 (SB); RFI has data, no switch.
        state.set_seat(1);
        assert_eq!(state.current_scenario, PreflopScenario::Rfi);

        // Switch to seat 0 (BB); RFI is empty, should land on vs_open.
        state.set_seat(0);
        assert_eq!(state.current_scenario, PreflopScenario::VsOpen);
    }

    #[test]
    fn set_seat_keeps_current_scenario_when_it_has_data() {
        use rs_poker::arena::cfr::PositionCharts;

        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        let mut position = PositionCharts::default();
        position.vs_open.set(aa, PreflopStrategy::pure_call());
        let config = PreflopChartConfig::with_single_position(position);
        let mut state = ChartViewerState::new(config, "test".into(), 6, false);

        // State starts on VsOpen (auto-picked since RFI is empty).
        assert_eq!(state.current_scenario, PreflopScenario::VsOpen);
        // Cycling seats keeps VsOpen — no reason to switch away.
        state.set_seat(3);
        assert_eq!(state.current_scenario, PreflopScenario::VsOpen);
    }
}
