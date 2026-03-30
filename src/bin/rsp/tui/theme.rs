use ratatui::{
    style::{Color, Modifier, Style},
    widgets::{Block, BorderType, Padding},
};

use crate::tui::state::PROFIT_EPSILON;

// ── Catppuccin Mocha Palette ──────────────────────────────────────────
// A soothing pastel theme. See https://github.com/catppuccin/catppuccin

// Accent colors
pub const FLAMINGO: Color = Color::Rgb(242, 205, 205);
pub const PINK: Color = Color::Rgb(245, 194, 231);
pub const MAUVE: Color = Color::Rgb(203, 166, 247);
pub const RED: Color = Color::Rgb(243, 139, 168);
pub const PEACH: Color = Color::Rgb(250, 179, 135);
pub const YELLOW: Color = Color::Rgb(249, 226, 175);
pub const GREEN: Color = Color::Rgb(166, 227, 161);
pub const TEAL: Color = Color::Rgb(148, 226, 213);
pub const SKY: Color = Color::Rgb(137, 220, 235);
pub const BLUE: Color = Color::Rgb(137, 180, 250);
pub const LAVENDER: Color = Color::Rgb(180, 190, 254);

// Neutral tones
pub const TEXT: Color = Color::Rgb(205, 214, 244);
pub const SUBTEXT1: Color = Color::Rgb(186, 194, 222);
pub const SUBTEXT0: Color = Color::Rgb(166, 173, 200);
pub const OVERLAY1: Color = Color::Rgb(127, 132, 156);
pub const OVERLAY0: Color = Color::Rgb(108, 112, 134);
pub const SURFACE2: Color = Color::Rgb(88, 91, 112);
pub const SURFACE1: Color = Color::Rgb(69, 71, 90);

// ── Semantic Colors ───────────────────────────────────────────────────

pub const PROFIT_COLOR: Color = GREEN;
pub const LOSS_COLOR: Color = RED;
pub const BREAKEVEN_COLOR: Color = YELLOW;
pub const HEADER_COLOR: Color = LAVENDER;
pub const STREET_COLOR: Color = MAUVE;
pub const BORDER_COLOR: Color = SURFACE2;
pub const FOCUS_COLOR: Color = LAVENDER;

// ── Panel Title Icons ─────────────────────────────────────────────────
// Card suit icons give a poker feel; widely supported Unicode.

pub const ICON_RANKINGS: &str = "♠";
pub const ICON_PROFIT: &str = "♦";
pub const ICON_STREET: &str = "♣";
pub const ICON_GAMES: &str = "♥";
pub const ICON_FILTER: &str = "⚙";
pub const ICON_AWARD: &str = "★";

// ── Agent Color Palette ───────────────────────────────────────────────
// Distinct pastel hues that are easy to distinguish.

pub const AGENT_COLORS: [Color; 8] = [BLUE, GREEN, PEACH, MAUVE, SKY, PINK, FLAMINGO, TEAL];

/// Return the color for agent at the given index, cycling through the palette.
pub fn agent_color(idx: usize) -> Color {
    AGENT_COLORS[idx % AGENT_COLORS.len()]
}

// ── Style Helpers ─────────────────────────────────────────────────────

/// Return a Style appropriate for the given profit value.
pub fn profit_style(profit: f32) -> Style {
    if profit > PROFIT_EPSILON {
        Style::default().fg(PROFIT_COLOR)
    } else if profit < -PROFIT_EPSILON {
        Style::default().fg(LOSS_COLOR)
    } else {
        Style::default().fg(BREAKEVEN_COLOR)
    }
}

/// Style for table/section headers.
pub fn header_style() -> Style {
    Style::default()
        .fg(HEADER_COLOR)
        .add_modifier(Modifier::BOLD)
}

/// Style for street labels.
pub fn street_style() -> Style {
    Style::default()
        .fg(STREET_COLOR)
        .add_modifier(Modifier::BOLD)
}

/// Style for the keybinding key itself (the part in brackets).
pub fn keybinding_key_style() -> Style {
    Style::default().fg(SUBTEXT0)
}

/// Create a bordered block for a panel, with focus styling and an icon.
pub fn panel_block(title: &str, focused: bool) -> Block<'static> {
    let border = if focused {
        Style::default()
            .fg(FOCUS_COLOR)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(BORDER_COLOR)
    };

    let title_style = if focused {
        Style::default().fg(LAVENDER).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(SUBTEXT1).add_modifier(Modifier::BOLD)
    };

    Block::bordered()
        .border_type(BorderType::Rounded)
        .title(format!(" {} ", title))
        .title_style(title_style)
        .border_style(border)
        .padding(Padding::horizontal(1))
}

/// Create a bordered block without padding (for charts/gauges that need all the space).
pub fn chart_block(title: &str) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .title(format!(" {} ", title))
        .title_style(Style::default().fg(SUBTEXT1).add_modifier(Modifier::BOLD))
        .border_style(Style::default().fg(BORDER_COLOR))
}

// ── Street Colors ─────────────────────────────────────────────────────
// A cool-to-warm gradient across the streets.

pub const STREET_PREFLOP: Color = MAUVE;
pub const STREET_FLOP: Color = BLUE;
pub const STREET_TURN: Color = TEAL;
pub const STREET_RIVER: Color = PEACH;
pub const STREET_SHOWDOWN: Color = PINK;

// ── Sort Indicator ────────────────────────────────────────────────────

pub const SORT_INDICATOR: &str = " ▲";

// ── Filter Symbols ────────────────────────────────────────────────────

pub const CHECK_ON: &str = "●";
pub const CHECK_OFF: &str = "○";
