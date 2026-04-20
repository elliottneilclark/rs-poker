//! 13x13 preflop hand grid widget.
//!
//! Renders a colored grid where each cell's background color encodes the
//! blended action mix for a `PreflopHand` in the chart passed in. Rows and
//! columns are labeled by rank (A → 2). The upper-right triangle (col > row)
//! is suited, the lower-left is offsuit, and the diagonal is pairs.
//!
//! A `PreflopChart` is scoped to a single (position, scenario) decision
//! point, so rendering is scenario-free — the caller picks which chart to
//! pass in.

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::Paragraph,
};

use rs_poker::core::Value;
use rs_poker::holdem::{PreflopActionType, PreflopChart, PreflopHand, PreflopStrategy};

use crate::tui::theme;

/// Number of combos for each cell type. Used when combo-weighting seat totals
/// so the bars read "X% of your dealt hands are a raise" rather than
/// "X% of the 169 grid cells".
const PAIR_COMBOS: f32 = 6.0;
const SUITED_COMBOS: f32 = 4.0;
const OFFSUIT_COMBOS: f32 = 12.0;

/// Ranks in display order, highest first (A at row 0 / col 0).
pub fn display_ranks() -> [Value; 13] {
    let mut ranks = Value::values();
    ranks.reverse();
    ranks
}

/// The `PreflopHand` shown at grid cell `(row, col)`.
pub fn hand_at(row: usize, col: usize) -> PreflopHand {
    let ranks = display_ranks();
    let high;
    let low;
    let suited;
    if row == col {
        high = ranks[row];
        low = ranks[row];
        suited = false;
    } else if col > row {
        // Upper-right: suited. Row rank is higher (smaller index = higher rank).
        high = ranks[row];
        low = ranks[col];
        suited = true;
    } else {
        // Lower-left: offsuit.
        high = ranks[col];
        low = ranks[row];
        suited = false;
    }
    PreflopHand::new(high, low, suited)
}

/// Combo count for a cell, used for range-weighted seat totals.
pub fn combo_count(row: usize, col: usize) -> f32 {
    if row == col {
        PAIR_COMBOS
    } else if col > row {
        SUITED_COMBOS
    } else {
        OFFSUIT_COMBOS
    }
}

/// Base color for each action type.
pub fn action_color(action: PreflopActionType) -> Color {
    match action {
        PreflopActionType::Fold => theme::SURFACE1,
        PreflopActionType::Call => theme::GREEN,
        PreflopActionType::Raise => theme::RED,
    }
}

/// Human-readable label for an action, used in legends and hover detail.
pub fn action_label(action: PreflopActionType) -> &'static str {
    match action {
        PreflopActionType::Fold => "Fold",
        PreflopActionType::Call => "Call",
        PreflopActionType::Raise => "Raise",
    }
}

/// Blend the raise/call/fold frequencies in `strategy` into a single
/// background color. `None` means the hand isn't in the chart — pure fold.
pub fn blended_color(strategy: Option<&PreflopStrategy>) -> Color {
    let (raise, call, fold) = match strategy {
        Some(s) => (s.raise(), s.call(), s.fold_freq()),
        None => return action_color(PreflopActionType::Fold),
    };
    let total = raise + call + fold;
    if total <= 0.0 {
        return action_color(PreflopActionType::Fold);
    }
    let (rr, rg, rb) = rgb_components(action_color(PreflopActionType::Raise));
    let (cr, cg, cb) = rgb_components(action_color(PreflopActionType::Call));
    let (fr, fg, fb) = rgb_components(action_color(PreflopActionType::Fold));
    let r = (rr * raise + cr * call + fr * fold) / total;
    let g = (rg * raise + cg * call + fg * fold) / total;
    let b = (rb * raise + cb * call + fb * fold) / total;
    Color::Rgb(
        r.round().clamp(0.0, 255.0) as u8,
        g.round().clamp(0.0, 255.0) as u8,
        b.round().clamp(0.0, 255.0) as u8,
    )
}

/// Pick a readable foreground color for text drawn on top of `bg`.
fn foreground_on(bg: Color) -> Color {
    let (r, g, b) = rgb_components(bg);
    // Rec. 709 luma. Below mid-gray → light text, else dark text.
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    if luma < 140.0 {
        theme::TEXT
    } else {
        theme::SURFACE1
    }
}

fn rgb_components(color: Color) -> (f32, f32, f32) {
    match color {
        Color::Rgb(r, g, b) => (r as f32, g as f32, b as f32),
        _ => (128.0, 128.0, 128.0),
    }
}

/// Range-weighted action totals for a chart.
///
/// Returns, for each of `[Fold, Call, Raise]`, the fraction of all dealt
/// starting hands that take that action at this seat/scenario. Pairs are
/// weighted by 6, suited by 4, and offsuit by 12, matching combo counts in a
/// 52-card deck (1326 total).
pub fn seat_totals(chart: &PreflopChart) -> [(PreflopActionType, f32); 3] {
    let mut raise = 0.0f32;
    let mut call = 0.0f32;
    let mut fold = 0.0f32;
    let mut total_weight = 0.0f32;

    for row in 0..13 {
        for col in 0..13 {
            let hand = hand_at(row, col);
            let weight = combo_count(row, col);
            total_weight += weight;
            match chart.get(&hand) {
                Some(strategy) => {
                    raise += strategy.raise() * weight;
                    call += strategy.call() * weight;
                    fold += strategy.fold_freq() * weight;
                }
                None => {
                    // Implicit fold for unlisted hands.
                    fold += weight;
                }
            }
        }
    }

    let norm = |v: f32| {
        if total_weight > 0.0 {
            v / total_weight
        } else {
            0.0
        }
    };
    [
        (PreflopActionType::Fold, norm(fold)),
        (PreflopActionType::Call, norm(call)),
        (PreflopActionType::Raise, norm(raise)),
    ]
}

/// Compute cell size that fills `area` while keeping cells visually square.
///
/// Terminal character cells are roughly 2:1 tall:wide in screen pixels, so
/// `cell_w = 2 * cell_h` renders as a visual square. Widths floor at 3 and
/// heights at 1 (the minimum 80-col layout). Widths cap at 9 so super-wide
/// terminals don't produce absurdly chunky cells.
///
/// Exposed so the event loop can use the same sizing when hit-testing
/// mouse clicks against the rendered grid.
pub fn cell_size(area: Rect) -> (u16, u16) {
    let avail_w = area.width.saturating_sub(2);
    let avail_h = area.height.saturating_sub(1);
    let cell_w = (avail_w / 13).clamp(3, 9);
    let cell_h = (avail_h / 13).max(1);
    let cell_h = cell_h.min((cell_w / 2).max(1));
    (cell_w, cell_h)
}

/// Render the 13x13 hand grid into `area` for `chart`.
///
/// The leftmost 2 columns hold the row axis labels (rank A–2) and the top
/// row holds the column axis labels.
pub fn render_hand_grid(
    frame: &mut Frame,
    area: Rect,
    chart: &PreflopChart,
    hover: Option<(usize, usize)>,
) {
    let (cell_w, cell_h) = cell_size(area);
    let ranks = display_ranks();

    // Column axis labels across the top (single-char rank, centered in cell).
    for (col, rank) in ranks.iter().enumerate() {
        let x = area.x + 2 + col as u16 * cell_w + cell_w / 2;
        let y = area.y;
        if x >= area.x + area.width || y >= area.y + area.height {
            continue;
        }
        let label = Paragraph::new(rank.to_char().to_string()).style(
            Style::default()
                .fg(theme::HEADER_COLOR)
                .add_modifier(Modifier::BOLD),
        );
        frame.render_widget(label, Rect::new(x, y, 1, 1));
    }

    // Row axis labels down the left, vertically centered in cell.
    for (row, rank) in ranks.iter().enumerate() {
        let x = area.x;
        let y = area.y + 1 + row as u16 * cell_h + cell_h / 2;
        if y >= area.y + area.height {
            continue;
        }
        let label = Paragraph::new(rank.to_char().to_string()).style(
            Style::default()
                .fg(theme::HEADER_COLOR)
                .add_modifier(Modifier::BOLD),
        );
        frame.render_widget(label, Rect::new(x, y, 1, 1));
    }

    for row in 0..13 {
        for col in 0..13 {
            let x = area.x + 2 + col as u16 * cell_w;
            let y = area.y + 1 + row as u16 * cell_h;
            if x + cell_w > area.x + area.width || y + cell_h > area.y + area.height {
                continue;
            }
            let hand = hand_at(row, col);
            let strategy = chart.get(&hand);
            let bg = blended_color(strategy);
            let is_hover = hover == Some((row, col));
            let fg = if is_hover {
                theme::LAVENDER
            } else {
                foreground_on(bg)
            };
            let mut style = Style::default().bg(bg).fg(fg);
            if is_hover {
                style = style.add_modifier(Modifier::BOLD);
            }

            // Fill cell_w x cell_h with background color. The hover dot (if
            // any) is centered in the middle row of the cell.
            let middle_row = cell_h / 2;
            for dy in 0..cell_h {
                let text = if is_hover && dy == middle_row {
                    let w = cell_w as usize;
                    let dot_idx = w / 2;
                    let mut s = String::with_capacity(w + 2);
                    s.extend(std::iter::repeat_n(' ', dot_idx));
                    s.push('●');
                    s.extend(std::iter::repeat_n(' ', w.saturating_sub(dot_idx + 1)));
                    s
                } else {
                    " ".repeat(cell_w as usize)
                };
                let cell = Paragraph::new(text).style(style);
                frame.render_widget(cell, Rect::new(x, y + dy, cell_w, 1));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_at_corners() {
        let aa = hand_at(0, 0);
        assert!(aa.is_pair());
        assert_eq!(aa.high(), Value::Ace);

        let a2s = hand_at(0, 12);
        assert!(a2s.suited());
        assert_eq!(a2s.high(), Value::Ace);
        assert_eq!(a2s.low(), Value::Two);

        let a2o = hand_at(12, 0);
        assert!(!a2o.is_pair());
        assert!(!a2o.suited());
        assert_eq!(a2o.high(), Value::Ace);
        assert_eq!(a2o.low(), Value::Two);

        let deuces = hand_at(12, 12);
        assert!(deuces.is_pair());
        assert_eq!(deuces.high(), Value::Two);
    }

    #[test]
    fn hand_at_standard_positions() {
        let aks = hand_at(0, 1);
        assert!(aks.suited());
        assert_eq!(aks.high(), Value::Ace);
        assert_eq!(aks.low(), Value::King);

        let ako = hand_at(1, 0);
        assert!(!ako.is_pair());
        assert!(!ako.suited());
        assert_eq!(ako.high(), Value::Ace);
        assert_eq!(ako.low(), Value::King);
    }

    #[test]
    fn combo_counts_match_convention() {
        assert_eq!(combo_count(0, 0), PAIR_COMBOS);
        assert_eq!(combo_count(0, 1), SUITED_COMBOS);
        assert_eq!(combo_count(1, 0), OFFSUIT_COMBOS);

        let mut total = 0.0;
        for r in 0..13 {
            for c in 0..13 {
                total += combo_count(r, c);
            }
        }
        assert_eq!(total as u32, 1326);
    }

    #[test]
    fn blended_color_pure_fold_for_missing_hand() {
        let color = blended_color(None);
        assert_eq!(color, theme::SURFACE1);
    }

    #[test]
    fn blended_color_pure_raise_matches_raise_color() {
        let strategy = PreflopStrategy::pure_raise();
        let color = blended_color(Some(&strategy));
        assert_eq!(color, theme::RED);
    }

    #[test]
    fn blended_color_mixes_equal_call_raise_to_midpoint() {
        let strategy = PreflopStrategy::new(0.5, 0.5).unwrap();
        let color = blended_color(Some(&strategy));
        let (rr, rg, rb) = rgb_components(theme::RED);
        let (gr, gg, gb) = rgb_components(theme::GREEN);
        let expected = Color::Rgb(
            ((rr + gr) / 2.0).round() as u8,
            ((rg + gg) / 2.0).round() as u8,
            ((rb + gb) / 2.0).round() as u8,
        );
        assert_eq!(color, expected);
    }

    #[test]
    fn seat_totals_all_fold_for_empty_chart() {
        let chart = PreflopChart::new();
        let totals = seat_totals(&chart);
        assert_eq!(totals[0].0, PreflopActionType::Fold);
        assert!((totals[0].1 - 1.0).abs() < 1e-5);
        assert!(totals[1].1.abs() < 1e-5);
        assert!(totals[2].1.abs() < 1e-5);
    }

    #[test]
    fn seat_totals_all_raise_for_every_hand() {
        let mut chart = PreflopChart::new();
        for hand in PreflopHand::all() {
            chart.set(hand, PreflopStrategy::pure_raise());
        }
        let totals = seat_totals(&chart);
        let raise = totals
            .iter()
            .find(|(a, _)| *a == PreflopActionType::Raise)
            .unwrap();
        assert!((raise.1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn render_grid_does_not_panic() {
        use ratatui::{Terminal, backend::TestBackend};
        let backend = TestBackend::new(60, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut chart = PreflopChart::new();
        chart.set(
            PreflopHand::new(Value::Ace, Value::Ace, false),
            PreflopStrategy::pure_raise(),
        );
        terminal
            .draw(|frame| {
                render_hand_grid(frame, Rect::new(0, 0, 50, 16), &chart, Some((0, 0)));
            })
            .unwrap();
    }

    #[test]
    fn cell_size_minimum_80col_layout() {
        let (w, h) = cell_size(Rect::new(0, 0, 41, 14));
        assert_eq!(w, 3);
        assert_eq!(h, 1);
    }

    #[test]
    fn cell_size_grows_width_on_wider_area() {
        let (w, _) = cell_size(Rect::new(0, 0, 80, 20));
        assert!(w > 3, "expected wider cells when area grows, got {}", w);
    }

    #[test]
    fn cell_size_grows_height_proportionally() {
        let (w, h) = cell_size(Rect::new(0, 0, 130, 40));
        assert!(w >= 5);
        assert!(h >= 2);
        assert!(h <= w / 2);
    }

    #[test]
    fn cell_size_caps_width() {
        let (w, _) = cell_size(Rect::new(0, 0, 400, 400));
        assert!(w <= 9, "cell width should cap, got {}", w);
    }

    #[test]
    fn render_grid_at_large_size_does_not_panic() {
        use ratatui::{Terminal, backend::TestBackend};
        let backend = TestBackend::new(200, 60);
        let mut terminal = Terminal::new(backend).unwrap();
        let chart = PreflopChart::new();
        terminal
            .draw(|frame| {
                render_hand_grid(frame, Rect::new(0, 0, 180, 55), &chart, Some((5, 7)));
            })
            .unwrap();
    }
}
