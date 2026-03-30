use std::time::Duration;

use ratatui::style::Color;
use tachyonfx::{CellFilter, Effect, EffectTimer, Interpolation, fx};

use crate::tui::theme::{FOCUS_COLOR, SURFACE2};

/// Border chase length as a fraction of perimeter (the bright "comet" tail).
const CHASE_LEN: f32 = 0.35;

/// Create a border chase effect: a bright comet runs around the border perimeter
/// and fades into the normal focus color.
pub fn border_chase() -> Effect {
    let timer = EffectTimer::new(Duration::from_millis(1000), Interpolation::Linear);

    fx::effect_fn_buf((), timer, |_state, ctx, buf| {
        let area = ctx.area;
        if area.width < 2 || area.height < 2 {
            return;
        }

        // Progress 0→1 over the animation
        let progress = ctx.alpha();

        // Build the perimeter path: top → right → bottom → left
        let w = area.width as usize;
        let h = area.height as usize;
        let perimeter = 2 * (w + h) - 4;
        if perimeter == 0 {
            return;
        }

        // The comet head position along the perimeter
        let head = (progress * perimeter as f32) as usize;

        // For each border cell, compute its position along the perimeter
        // and set its fg color based on distance from the comet head.
        for i in 0..perimeter {
            let (x, y) = perimeter_pos(area.x, area.y, w, h, i);
            let cell = &mut buf[(x, y)];

            let tail_len = CHASE_LEN * perimeter as f32;
            // How far behind the comet head this cell is (wrapping)
            let behind = (head as isize - i as isize).rem_euclid(perimeter as isize) as f32;

            if behind < tail_len {
                // In the comet tail: bright at head, fading to focus color
                let t = behind / tail_len;
                let color = lerp_color(Color::White, FOCUS_COLOR, t);
                cell.set_fg(color);
            } else {
                // Outside the comet: fade from dim border to focus color
                let color = lerp_color(SURFACE2, FOCUS_COLOR, progress);
                cell.set_fg(color);
            }
        }
    })
    .with_filter(CellFilter::Outer(ratatui::layout::Margin::new(1, 1)))
}

/// Map a perimeter index to (x, y) coordinates for a rect.
/// Path: top-left → top-right → bottom-right → bottom-left → back.
fn perimeter_pos(rx: u16, ry: u16, w: usize, h: usize, idx: usize) -> (u16, u16) {
    let top = w;
    let right = top + h - 1;
    let bottom = right + w - 1;

    if idx < top {
        // Top edge: left to right
        (rx + idx as u16, ry)
    } else if idx < right {
        // Right edge: top to bottom
        let offset = idx - top + 1;
        (rx + w as u16 - 1, ry + offset as u16)
    } else if idx < bottom {
        // Bottom edge: right to left
        let offset = idx - right + 1;
        (rx + w as u16 - 1 - offset as u16, ry + h as u16 - 1)
    } else {
        // Left edge: bottom to top
        let offset = idx - bottom + 1;
        (rx, ry + h as u16 - 1 - offset as u16)
    }
}

/// Linearly interpolate between two RGB colors. Falls back to `to` for non-RGB.
fn lerp_color(from: Color, to: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    match (from, to) {
        (Color::Rgb(r1, g1, b1), Color::Rgb(r2, g2, b2)) => {
            let r = (r1 as f32 + (r2 as f32 - r1 as f32) * t) as u8;
            let g = (g1 as f32 + (g2 as f32 - g1 as f32) * t) as u8;
            let b = (b1 as f32 + (b2 as f32 - b1 as f32) * t) as u8;
            Color::Rgb(r, g, b)
        }
        _ => to,
    }
}
