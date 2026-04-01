use ratatui::{
    Frame,
    layout::Rect,
    style::Style,
    widgets::{Cell, Row, Table},
};

use crate::tui::{
    state::AgentDisplayData,
    theme::{
        self, ICON_RANKINGS, OVERLAY0, SORT_INDICATOR, SURFACE1, TEXT, header_style, panel_block,
        profit_style,
    },
};

/// Column identifiers for sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortColumn {
    Name,
    Profit,
    Games,
    WinPct,
    Roi,
    Vpip,
    Pfr,
    ThreeBet,
    Af,
    Cbet,
    Wtsd,
    Wsd,
}

impl SortColumn {
    pub fn next(self) -> Self {
        match self {
            Self::Name => Self::Profit,
            Self::Profit => Self::Games,
            Self::Games => Self::WinPct,
            Self::WinPct => Self::Roi,
            Self::Roi => Self::Vpip,
            Self::Vpip => Self::Pfr,
            Self::Pfr => Self::ThreeBet,
            Self::ThreeBet => Self::Af,
            Self::Af => Self::Cbet,
            Self::Cbet => Self::Wtsd,
            Self::Wtsd => Self::Wsd,
            Self::Wsd => Self::Name,
        }
    }

    fn header(&self) -> &'static str {
        match self {
            Self::Name => "Agent",
            Self::Profit => "Profit(bb)",
            Self::Games => "Games",
            Self::WinPct => "Win%",
            Self::Roi => "ROI%",
            Self::Vpip => "VPIP%",
            Self::Pfr => "PFR%",
            Self::ThreeBet => "3Bet%",
            Self::Af => "AF",
            Self::Cbet => "CBet%",
            Self::Wtsd => "WTSD%",
            Self::Wsd => "W$SD%",
        }
    }
}

const ALL_COLUMNS: [SortColumn; 12] = [
    SortColumn::Name,
    SortColumn::Profit,
    SortColumn::Games,
    SortColumn::WinPct,
    SortColumn::Roi,
    SortColumn::Vpip,
    SortColumn::Pfr,
    SortColumn::ThreeBet,
    SortColumn::Af,
    SortColumn::Cbet,
    SortColumn::Wtsd,
    SortColumn::Wsd,
];

pub fn render_stats_table(
    frame: &mut Frame,
    area: Rect,
    agents: &[AgentDisplayData],
    selected: Option<usize>,
    sort_col: SortColumn,
    focused: bool,
) {
    let header_cells: Vec<Cell> = ALL_COLUMNS
        .iter()
        .map(|col| {
            let label = if *col == sort_col {
                format!("{}{}", col.header(), SORT_INDICATOR)
            } else {
                col.header().to_string()
            };
            Cell::from(label).style(header_style())
        })
        .collect();
    let header = Row::new(header_cells).height(1).bottom_margin(1);

    let rows: Vec<Row> = agents
        .iter()
        .enumerate()
        .map(|(idx, agent)| {
            let win_pct = if agent.games_played > 0 {
                agent.wins as f32 / agent.games_played as f32 * 100.0
            } else {
                0.0
            };

            let cells = vec![
                Cell::from(agent.name.clone()).style(Style::default().fg(theme::agent_color(idx))),
                Cell::from(format!("{:+.1}", agent.profit_bb)).style(profit_style(agent.profit_bb)),
                Cell::from(format!("{}", agent.games_played)).style(Style::default().fg(TEXT)),
                Cell::from(format!("{:.1}", win_pct)).style(Style::default().fg(TEXT)),
                Cell::from(format!("{:.1}", agent.roi_percent)).style(Style::default().fg(TEXT)),
                Cell::from(format!("{:.1}", agent.vpip_percent))
                    .style(Style::default().fg(OVERLAY0)),
                Cell::from(format!("{:.1}", agent.pfr_percent))
                    .style(Style::default().fg(OVERLAY0)),
                Cell::from(format!("{:.1}", agent.three_bet_percent))
                    .style(Style::default().fg(OVERLAY0)),
                Cell::from(format!("{:.2}", agent.aggression_factor))
                    .style(Style::default().fg(OVERLAY0)),
                Cell::from(format!("{:.1}", agent.cbet_percent))
                    .style(Style::default().fg(OVERLAY0)),
                Cell::from(format!("{:.1}", agent.wtsd_percent))
                    .style(Style::default().fg(OVERLAY0)),
                Cell::from(format!("{:.1}", agent.wsd_percent))
                    .style(Style::default().fg(OVERLAY0)),
            ];

            Row::new(cells)
        })
        .collect();

    let widths = [
        ratatui::layout::Constraint::Min(12),
        ratatui::layout::Constraint::Length(10),
        ratatui::layout::Constraint::Length(7),
        ratatui::layout::Constraint::Length(6),
        ratatui::layout::Constraint::Length(7),
        ratatui::layout::Constraint::Length(7),
        ratatui::layout::Constraint::Length(6),
        ratatui::layout::Constraint::Length(6),
        ratatui::layout::Constraint::Length(6),
        ratatui::layout::Constraint::Length(7),
        ratatui::layout::Constraint::Length(7),
        ratatui::layout::Constraint::Length(7),
    ];

    let title = format!("{} Agent Rankings", ICON_RANKINGS);
    let table = Table::new(rows, widths)
        .header(header)
        .block(panel_block(&title, focused))
        .row_highlight_style(Style::default().bg(SURFACE1).fg(TEXT));

    if let Some(sel) = selected {
        let mut table_state = ratatui::widgets::TableState::default();
        table_state.select(Some(sel));
        frame.render_stateful_widget(table, area, &mut table_state);
    } else {
        frame.render_widget(table, area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use ratatui::{Terminal, backend::TestBackend};

    fn make_agent(name: &str, profit: f32, games: usize, wins: usize) -> AgentDisplayData {
        AgentDisplayData {
            name: name.to_string(),
            total_profit: profit,
            profit_bb: profit / 10.0,
            games_played: games,
            wins,
            vpip_percent: 25.0,
            pfr_percent: 18.0,
            three_bet_percent: 8.0,
            aggression_factor: 2.5,
            cbet_percent: 60.0,
            wtsd_percent: 30.0,
            wsd_percent: 55.0,
            roi_percent: 5.0,
        }
    }

    #[test]
    fn test_render_stats_table() {
        let backend = TestBackend::new(120, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let agents = vec![
            make_agent("Alice", 150.0, 100, 55),
            make_agent("Bob", -50.0, 100, 45),
        ];
        terminal
            .draw(|frame| {
                render_stats_table(frame, frame.area(), &agents, None, SortColumn::Profit, true);
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }

    #[test]
    fn test_render_stats_table_with_selection() {
        let backend = TestBackend::new(120, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let agents = vec![
            make_agent("Alice", 150.0, 100, 55),
            make_agent("Bob", -50.0, 100, 45),
        ];
        terminal
            .draw(|frame| {
                render_stats_table(
                    frame,
                    frame.area(),
                    &agents,
                    Some(0),
                    SortColumn::Profit,
                    true,
                );
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }

    #[test]
    fn test_render_stats_table_empty() {
        let backend = TestBackend::new(120, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render_stats_table(frame, frame.area(), &[], None, SortColumn::Profit, false);
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }
}
