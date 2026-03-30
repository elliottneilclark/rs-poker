use ratatui::{
    Frame,
    layout::Rect,
    style::Style,
    text::Line,
    widgets::{Bar, BarChart, BarGroup},
};

use crate::tui::{
    state::StreetDistribution,
    theme::{
        ICON_STREET, STREET_FLOP, STREET_PREFLOP, STREET_RIVER, STREET_SHOWDOWN, STREET_TURN, TEXT,
        chart_block,
    },
};

pub fn render_street_bars(frame: &mut Frame, area: Rect, dist: &StreetDistribution) {
    let total = dist.total() as f64;
    let pct = |v: usize| -> u64 {
        if total > 0.0 {
            (v as f64 / total * 100.0) as u64
        } else {
            0
        }
    };

    let bars = vec![
        Bar::default()
            .label(Line::from("Preflop"))
            .value(pct(dist.preflop))
            .style(Style::default().fg(STREET_PREFLOP)),
        Bar::default()
            .label(Line::from("Flop"))
            .value(pct(dist.flop))
            .style(Style::default().fg(STREET_FLOP)),
        Bar::default()
            .label(Line::from("Turn"))
            .value(pct(dist.turn))
            .style(Style::default().fg(STREET_TURN)),
        Bar::default()
            .label(Line::from("River"))
            .value(pct(dist.river))
            .style(Style::default().fg(STREET_RIVER)),
        Bar::default()
            .label(Line::from("Showdown"))
            .value(pct(dist.showdown))
            .style(Style::default().fg(STREET_SHOWDOWN)),
    ];

    let title = format!("{} Street Distribution", ICON_STREET);
    let bar_chart = BarChart::default()
        .block(chart_block(&title))
        .data(BarGroup::default().bars(&bars))
        .bar_width(8)
        .bar_gap(1)
        .value_style(Style::default().fg(TEXT));

    frame.render_widget(bar_chart, area);
}
