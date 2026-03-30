use std::collections::HashMap;

use ratatui::{
    Frame,
    layout::Rect,
    style::Style,
    symbols::Marker,
    widgets::{Axis, Chart, Dataset, GraphType},
};

use crate::tui::{
    state::AgentDisplayData,
    theme::{self, ICON_PROFIT, SUBTEXT0, chart_block},
};

pub fn render_profit_chart(
    frame: &mut Frame,
    area: Rect,
    agents: &[AgentDisplayData],
    profit_histories: &HashMap<String, Vec<f32>>,
) {
    let title = format!("{} Profit Over Time", ICON_PROFIT);

    if agents.is_empty() {
        frame.render_widget(chart_block(&title), area);
        return;
    }

    let empty = Vec::new();

    // Pre-compute data points for each agent
    let chart_data: Vec<Vec<(f64, f64)>> = agents
        .iter()
        .map(|agent| {
            let history = profit_histories.get(&agent.name).unwrap_or(&empty);
            history
                .iter()
                .enumerate()
                .map(|(i, &p)| (i as f64, p as f64))
                .collect()
        })
        .collect();

    // Find bounds
    let max_x = chart_data.iter().map(|d| d.len()).max().unwrap_or(1) as f64;
    let (min_y, max_y) = chart_data
        .iter()
        .flat_map(|d| d.iter())
        .fold((0.0_f64, 0.0_f64), |(min, max), &(_, y)| {
            (min.min(y), max.max(y))
        });
    let y_margin = (max_y - min_y).max(10.0) * 0.1;

    let datasets: Vec<Dataset> = agents
        .iter()
        .enumerate()
        .map(|(i, agent)| {
            Dataset::default()
                .name(agent.name.as_str())
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(theme::agent_color(i)))
                .data(&chart_data[i])
        })
        .collect();

    let chart = Chart::new(datasets)
        .block(chart_block(&title))
        .x_axis(
            Axis::default()
                .bounds([0.0, max_x])
                .title("Game")
                .style(Style::default().fg(SUBTEXT0)),
        )
        .y_axis(
            Axis::default()
                .bounds([min_y - y_margin, max_y + y_margin])
                .title("Profit")
                .style(Style::default().fg(SUBTEXT0)),
        );

    frame.render_widget(chart, area);
}
