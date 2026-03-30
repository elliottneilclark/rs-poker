use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Gauge, Paragraph},
};

use crate::tui::{
    state::TuiState,
    theme::{LAVENDER, OVERLAY0, RED, SKY, SUBTEXT0, SURFACE1, TEXT, keybinding_key_style},
};

pub fn render_progress(frame: &mut Frame, area: Rect, state: &TuiState) {
    if let Some(ref err) = state.error {
        let line = Line::from(Span::styled(
            format!(" ✕ Error: {} ", err),
            Style::default().fg(RED),
        ));
        frame.render_widget(Paragraph::new(line), area);
        return;
    }
    if state.live {
        render_live_progress(frame, area, state);
    } else {
        render_static_status(frame, area, state);
    }
}

fn keybinding_hints() -> Vec<Span<'static>> {
    vec![
        Span::styled("q", keybinding_key_style()),
        Span::styled(" Quit  ", Style::default().fg(OVERLAY0)),
        Span::styled("j/k", keybinding_key_style()),
        Span::styled(" ↕  ", Style::default().fg(OVERLAY0)),
        Span::styled("^d/^u", keybinding_key_style()),
        Span::styled(" ½Pg  ", Style::default().fg(OVERLAY0)),
        Span::styled("g/G", keybinding_key_style()),
        Span::styled(" Top/End  ", Style::default().fg(OVERLAY0)),
        Span::styled("Tab", keybinding_key_style()),
        Span::styled(" Panel  ", Style::default().fg(OVERLAY0)),
        Span::styled("Enter", keybinding_key_style()),
        Span::styled(" Select  ", Style::default().fg(OVERLAY0)),
        Span::styled("s", keybinding_key_style()),
        Span::styled(" Sort", Style::default().fg(OVERLAY0)),
    ]
}

fn render_live_progress(frame: &mut Frame, area: Rect, state: &TuiState) {
    let chunks =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(area);

    // Left side: progress gauge or counter
    if let Some(target) = state.games_target {
        let ratio = if target > 0 {
            state.games_completed as f64 / target as f64
        } else {
            0.0
        };
        let gauge = Gauge::default()
            .ratio(ratio.min(1.0))
            .label(format!("{} / {}", state.games_completed, target))
            .gauge_style(Style::default().fg(LAVENDER).bg(SURFACE1));
        frame.render_widget(gauge, chunks[0]);
    } else {
        let counter = Paragraph::new(Span::styled(
            format!("{} games", state.games_completed),
            Style::default().fg(TEXT),
        ));
        frame.render_widget(counter, chunks[0]);
    };

    // Right side: stats and keybindings
    let gps = state.games_per_second();
    let elapsed = state.elapsed();
    let elapsed_str = format!("{}:{:02}", elapsed.as_secs() / 60, elapsed.as_secs() % 60);

    let eta_str = state
        .eta()
        .map(|d| format!("{}:{:02}", d.as_secs() / 60, d.as_secs() % 60))
        .unwrap_or_else(|| "--:--".to_string());

    let mut spans = vec![
        Span::styled(format!(" {:.0} g/s", gps), Style::default().fg(SKY)),
        Span::styled(
            format!(" │ {} │ ETA {} │ ", elapsed_str, eta_str),
            Style::default().fg(SUBTEXT0),
        ),
    ];
    spans.extend(keybinding_hints());

    frame.render_widget(Paragraph::new(Line::from(spans)), chunks[1]);
}

fn render_static_status(frame: &mut Frame, area: Rect, state: &TuiState) {
    let mut spans = vec![
        Span::styled(
            format!(" {} games loaded ", state.games_completed),
            Style::default().fg(SKY),
        ),
        Span::styled("│ ", Style::default().fg(SUBTEXT0)),
    ];
    spans.extend(keybinding_hints());

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn test_render_static_status() {
        let backend = TestBackend::new(100, 1);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = TuiState::new(Some(50));
        state.live = false;
        state.games_completed = 42;
        terminal
            .draw(|frame| {
                render_progress(frame, frame.area(), &state);
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }

    #[test]
    fn test_render_error_status() {
        let backend = TestBackend::new(100, 1);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = TuiState::new(Some(50));
        state.error = Some(crate::tui::event::SimError::Panic);
        terminal
            .draw(|frame| {
                render_progress(frame, frame.area(), &state);
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }
}
