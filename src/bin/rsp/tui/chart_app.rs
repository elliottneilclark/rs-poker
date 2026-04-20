//! Standalone TUI app for viewing preflop charts.
//!
//! Unlike [`crate::tui::app::App`], which drives a live simulation and merges
//! a bounded sync_channel of `SimMessage`s with crossterm events, this app
//! only reacts to keyboard/mouse input. There is no background thread and
//! no tick-based refresh beyond a short polling interval.

use std::time::Duration;

use crossterm::event::{
    self, Event as CrosstermEvent, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};

use rs_poker::holdem::PreflopScenario;

use crate::tui::{
    screens::chart_viewer::{ChartViewerRects, ChartViewerState, render_chart_viewer},
    terminal::{self, Tui},
    widgets::hand_grid::cell_size,
};

/// App owning a `ChartViewerState` plus input-handling flags.
pub struct ChartApp {
    pub state: ChartViewerState,
    pub should_quit: bool,
    rects: Option<ChartViewerRects>,
}

impl ChartApp {
    pub fn new(state: ChartViewerState) -> Self {
        Self {
            state,
            should_quit: false,
            rects: None,
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.should_quit = true;
            return;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Char('j') | KeyCode::Down => self.state.move_hover(1, 0),
            KeyCode::Char('k') | KeyCode::Up => self.state.move_hover(-1, 0),
            KeyCode::Char('h') | KeyCode::Left => self.state.move_hover(0, -1),
            KeyCode::Char('l') | KeyCode::Right => self.state.move_hover(0, 1),
            KeyCode::Tab => self.state.next_seat(),
            KeyCode::BackTab => self.state.prev_seat(),
            KeyCode::Char(']') => self.state.next_seat(),
            KeyCode::Char('[') => self.state.prev_seat(),
            // Scenario jumps. Digit keys conflict with seat jumps, so we use
            // letters: `r`=RFI, `o`=vs Open, `t`=vs 3-Bet, `f`=vs 4-Bet.
            // `s`/`S` also cycle scenarios forward/backward.
            KeyCode::Char('r') => self.state.set_scenario(PreflopScenario::Rfi),
            KeyCode::Char('o') => self.state.set_scenario(PreflopScenario::VsOpen),
            KeyCode::Char('t') => self.state.set_scenario(PreflopScenario::Vs3Bet),
            KeyCode::Char('f') => self.state.set_scenario(PreflopScenario::Vs4Bet),
            KeyCode::Char('s') => self.state.next_scenario(),
            KeyCode::Char('S') => self.state.prev_scenario(),
            KeyCode::Char(c) if c.is_ascii_digit() && c != '0' => {
                let seat = (c as u8 - b'1') as usize;
                if seat < self.state.num_seats {
                    self.state.set_seat(seat);
                }
            }
            _ => {}
        }
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) {
        let Some(rects) = &self.rects else {
            return;
        };
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                for (seat, rect) in rects.seat_tabs.iter().enumerate() {
                    if rect.contains((mouse.column, mouse.row).into()) {
                        self.state.set_seat(seat);
                        return;
                    }
                }
                for (scenario, rect) in &rects.scenario_tabs {
                    if rect.contains((mouse.column, mouse.row).into()) {
                        self.state.set_scenario(*scenario);
                        return;
                    }
                }
                if let Some((row, col)) = grid_hit(&rects.grid, mouse.column, mouse.row) {
                    self.state.hover = (row, col);
                }
            }
            // Drag: keep moving the hover as the user drags across the grid.
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some((row, col)) = grid_hit(&rects.grid, mouse.column, mouse.row) {
                    self.state.hover = (row, col);
                }
            }
            // Scroll wheel over the seat tab row cycles seats — a fast way to
            // compare positions without reaching for the keyboard.
            MouseEventKind::ScrollUp
                if rects
                    .seat_tabs
                    .iter()
                    .any(|r| r.contains((mouse.column, mouse.row).into())) =>
            {
                self.state.prev_seat();
            }
            MouseEventKind::ScrollDown
                if rects
                    .seat_tabs
                    .iter()
                    .any(|r| r.contains((mouse.column, mouse.row).into())) =>
            {
                self.state.next_seat();
            }
            _ => {}
        }
    }
}

/// Convert a mouse (column, row) position inside the grid area to a
/// `(row, col)` cell index. Returns `None` if the position landed outside
/// the 13x13 cell region (axis labels, gutters, or padding beyond the grid).
///
/// Uses [`cell_size`] so responsive cell dimensions stay consistent with
/// what [`render_hand_grid`](super::widgets::hand_grid::render_hand_grid)
/// drew.
fn grid_hit(grid: &ratatui::layout::Rect, col: u16, row: u16) -> Option<(usize, usize)> {
    let (cell_w, cell_h) = cell_size(*grid);
    if col < grid.x + 2 || row < grid.y + 1 {
        return None;
    }
    let c = ((col - grid.x - 2) / cell_w) as usize;
    let r = ((row - grid.y - 1) / cell_h) as usize;
    if r < 13 && c < 13 { Some((r, c)) } else { None }
}

/// Drive the chart viewer event loop until the user quits.
pub fn run_chart_app(app: &mut ChartApp) -> std::io::Result<()> {
    let mut terminal = terminal::setup_terminal()?;
    let result = event_loop(&mut terminal, app);
    terminal::restore_terminal(&mut terminal)?;
    result
}

fn event_loop(terminal: &mut Tui, app: &mut ChartApp) -> std::io::Result<()> {
    loop {
        let mut new_rects = None;
        terminal.draw(|frame| {
            new_rects = Some(render_chart_viewer(frame, &app.state));
        })?;
        app.rects = new_rects;

        if app.should_quit {
            break;
        }

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                CrosstermEvent::Key(key) => app.handle_key(key),
                CrosstermEvent::Mouse(mouse) => app.handle_mouse(mouse),
                CrosstermEvent::Resize(_, _) => {}
                _ => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;
    use rs_poker::arena::cfr::{PositionCharts, PreflopChartConfig};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn app() -> ChartApp {
        let config = PreflopChartConfig::with_single_position(PositionCharts::default());
        ChartApp::new(ChartViewerState::new(config, "test".into(), 6, false))
    }

    #[test]
    fn q_quits() {
        let mut a = app();
        a.handle_key(key(KeyCode::Char('q')));
        assert!(a.should_quit);
    }

    #[test]
    fn ctrl_c_quits() {
        let mut a = app();
        a.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(a.should_quit);
    }

    #[test]
    fn hjkl_moves_hover() {
        let mut a = app();
        a.state.hover = (5, 5);
        a.handle_key(key(KeyCode::Char('j')));
        assert_eq!(a.state.hover, (6, 5));
        a.handle_key(key(KeyCode::Char('k')));
        assert_eq!(a.state.hover, (5, 5));
        a.handle_key(key(KeyCode::Char('l')));
        assert_eq!(a.state.hover, (5, 6));
        a.handle_key(key(KeyCode::Char('h')));
        assert_eq!(a.state.hover, (5, 5));
    }

    #[test]
    fn digit_keys_jump_to_seat() {
        let mut a = app();
        a.handle_key(key(KeyCode::Char('3')));
        assert_eq!(a.state.current_seat, 2);
        a.handle_key(key(KeyCode::Char('1')));
        assert_eq!(a.state.current_seat, 0);
    }

    #[test]
    fn tab_cycles_seats() {
        let mut a = app();
        a.state.set_seat(5);
        a.handle_key(key(KeyCode::Tab));
        assert_eq!(a.state.current_seat, 0);
    }

    #[test]
    fn digit_outside_range_is_ignored() {
        let mut a = app();
        a.state.set_seat(2);
        // num_seats = 6, so '9' (seat 8) must not change anything.
        a.handle_key(key(KeyCode::Char('9')));
        assert_eq!(a.state.current_seat, 2);
    }

    #[test]
    fn grid_hit_inside_cell() {
        // 50 wide, 16 tall → cell_size returns (3, 1).
        let grid = ratatui::layout::Rect::new(0, 0, 50, 16);
        // Cell (0, 0) starts at col=2, row=1
        assert_eq!(grid_hit(&grid, 2, 1), Some((0, 0)));
        // Cell (0, 1) starts at col=5, row=1
        assert_eq!(grid_hit(&grid, 6, 1), Some((0, 1)));
        // Axis-label area returns None.
        assert_eq!(grid_hit(&grid, 0, 0), None);
        assert_eq!(grid_hit(&grid, 1, 1), None);
    }

    #[test]
    fn grid_hit_scales_with_responsive_cells() {
        // 130 wide, 40 tall → cells are wider and taller.
        // avail_w=128, cell_w=128/13=9 (capped); avail_h=39, cell_h=min(3, 9/2=4)=3.
        let grid = ratatui::layout::Rect::new(0, 0, 130, 40);
        let (cw, ch) = super::cell_size(grid);
        // Click in the middle of cell (2, 5) should resolve to (2, 5).
        let x = 2 + 5 * cw + cw / 2;
        let y = 1 + 2 * ch + ch / 2;
        assert_eq!(grid_hit(&grid, x, y), Some((2, 5)));
        // Click at the very start of the cells row/col.
        assert_eq!(grid_hit(&grid, 2, 1), Some((0, 0)));
    }
}
