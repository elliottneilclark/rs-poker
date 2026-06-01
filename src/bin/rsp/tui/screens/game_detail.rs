use std::collections::HashMap;

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Cell, Paragraph, Row, Table},
};

use rs_poker::{
    core::{Card, CoreRank, Rankable},
    open_hand_history::{Action, BetType, GameType, HandHistory},
};

use crate::tui::state::{PROFIT_EPSILON, compute_hand_profits};
use crate::tui::theme::{
    self, ICON_AWARD, ICON_GAMES, ICON_STREET, LAVENDER, MAUVE, OVERLAY0, PEACH, SKY, SUBTEXT0,
    SURFACE2, TEXT, YELLOW, header_style, keybinding_key_style, panel_block, profit_style,
    street_style,
};

pub fn render_detail(frame: &mut Frame, hand: &HandHistory, scroll: u16) {
    let area = frame.area();

    let info_height = 10_u16; // 8 rows + 2 border
    let players_height = hand.players.len() as u16 + 4; // rows + header + margin + 2 border
    let top_height = info_height.max(players_height);

    let chunks = Layout::vertical([
        Constraint::Length(top_height),
        Constraint::Min(10),
        Constraint::Length(1),
    ])
    .split(area);

    let top_cols = Layout::horizontal([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(chunks[0]);

    render_game_info(frame, top_cols[0], hand);
    render_players_table(frame, top_cols[1], hand);
    render_round_log(frame, chunks[1], hand, scroll);
    render_detail_keybindings(frame, chunks[2]);
}

fn render_game_info(frame: &mut Frame, area: Rect, hand: &HandHistory) {
    let block = panel_block(&format!("{} Game Info", theme::ICON_RANKINGS), false);

    let game_type = format_game_type(&hand.game_type);
    let limit = hand
        .bet_limit
        .as_ref()
        .map(|bl| format_bet_type(&bl.bet_type))
        .unwrap_or("—");

    let blinds = if hand.ante_amount > 0.0 {
        format!(
            "{:.0} / {:.0} (ante {:.0})",
            hand.small_blind_amount, hand.big_blind_amount, hand.ante_amount
        )
    } else {
        format!(
            "{:.0} / {:.0}",
            hand.small_blind_amount, hand.big_blind_amount
        )
    };

    let date = hand
        .start_date_utc
        .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|| "—".into());

    let table_label = format!("{} ({}-max)", hand.table_name, hand.table_size);

    let pairs: Vec<(&str, String)> = vec![
        ("Site", hand.site_name.clone()),
        ("Table", table_label),
        ("Game", game_type.into()),
        ("Limit", limit.into()),
        ("Blinds", blinds),
        ("Date", date),
        ("Game #", hand.game_number.clone()),
        ("Dealer", format!("Seat {}", hand.dealer_seat)),
    ];

    let lines: Vec<Line> = pairs
        .into_iter()
        .map(|(key, val)| {
            Line::from(vec![
                Span::styled(format!("{:<8} ", key), Style::default().fg(SUBTEXT0)),
                Span::styled(val, Style::default().fg(TEXT)),
            ])
        })
        .collect();

    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_players_table(frame: &mut Frame, area: Rect, hand: &HandHistory) {
    let block = panel_block(&format!("{} Players", ICON_GAMES), false);
    let inner = block.inner(area);

    let (_id_to_idx, profits) = compute_hand_profits(hand);

    let header = Row::new(vec!["Seat", "Name", "Start", "End", "Change", "Cards"])
        .style(header_style())
        .bottom_margin(1);

    let rows: Vec<Row> = hand
        .players
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            let is_dealer = p.seat == hand.dealer_seat;
            let seat_str = format!("{}", p.seat);

            let name_str = if is_dealer {
                format!("{} Ⓓ", p.name)
            } else {
                p.name.clone()
            };

            let name_style = Style::default().fg(theme::agent_color(idx));

            let profit = profits.get(idx).copied().unwrap_or(0.0);
            let profit_str = if profit > PROFIT_EPSILON {
                format!("+{:.1}", profit)
            } else {
                format!("{:.1}", profit)
            };

            // Ending stack = starting stack net of this hand's change
            // (change = winnings - invested, so end = start + change).
            let ending_stack = p.starting_stack + profit;

            let cards_str = hand
                .rounds
                .first()
                .and_then(|r| {
                    r.actions
                        .iter()
                        .find(|a| a.player_id == p.id && a.action == Action::DealtCards)
                })
                .and_then(|a| a.cards.as_ref())
                .map(|cards| {
                    cards
                        .iter()
                        .map(|c| c.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();

            Row::new(vec![
                Cell::from(seat_str).style(Style::default().fg(OVERLAY0)),
                Cell::from(name_str).style(name_style),
                Cell::from(format!("{:.0}", p.starting_stack)).style(Style::default().fg(TEXT)),
                Cell::from(format!("{:.0}", ending_stack)).style(Style::default().fg(TEXT)),
                Cell::from(profit_str).style(profit_style(profit)),
                Cell::from(cards_str).style(Style::default().fg(YELLOW)),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(5),
        Constraint::Min(10),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(8),
        Constraint::Length(12),
    ];

    let table = Table::new(rows, widths).header(header);
    frame.render_widget(block, area);
    frame.render_widget(table, inner);
}

fn render_round_log(frame: &mut Frame, area: Rect, hand: &HandHistory, scroll: u16) {
    let block = panel_block(&format!("{} Round Log", ICON_STREET), false);

    let mut lines: Vec<Line> = Vec::new();
    let mut running_pot: f32 = 0.0;
    let mut board: Vec<Card> = Vec::new();

    // Build player lookups: index for coloring, name for display
    let player_index: HashMap<u64, usize> = hand
        .players
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id, i))
        .collect();
    let player_names: HashMap<u64, &str> = hand
        .players
        .iter()
        .map(|p| (p.id, p.name.as_str()))
        .collect();

    for round in &hand.rounds {
        // Accumulate community cards
        if let Some(cards) = &round.cards {
            board.extend(cards.iter().copied());
        }

        // Per-street cumulative contribution per player. Used to report the
        // total bet a caller is matching (their running total this street),
        // not just the incremental chips they added.
        let mut street_contrib: HashMap<u64, f32> = HashMap::new();

        let board_str = round
            .cards
            .as_ref()
            .map(|cards| {
                let s = cards
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                format!(" [{}]", s)
            })
            .unwrap_or_default();

        lines.push(Line::from(vec![Span::styled(
            format!("── {} ──{}", round.street, board_str),
            street_style(),
        )]));

        for action in &round.actions {
            // Skip non-display actions
            if matches!(
                action.action,
                Action::DealtCards | Action::SitsDown | Action::StandsUp | Action::AddedChips
            ) {
                continue;
            }

            let player_name = player_names.get(&action.player_id).copied().unwrap_or("?");

            let player_idx = player_index.get(&action.player_id).copied().unwrap_or(0);

            let action_style = match action.action {
                Action::Fold => Style::default().fg(OVERLAY0),
                Action::Bet | Action::Raise => Style::default().fg(PEACH),
                Action::Call => Style::default().fg(SKY),
                Action::Check => Style::default().fg(SUBTEXT0),
                _ => Style::default().fg(OVERLAY0),
            };

            // Track running pot for betting actions
            if matches!(
                action.action,
                Action::Bet
                    | Action::Raise
                    | Action::Call
                    | Action::PostSmallBlind
                    | Action::PostBigBlind
                    | Action::PostAnte
                    | Action::Straddle
                    | Action::PostDead
                    | Action::PostExtraBlind
                    | Action::AddedToPot
            ) {
                running_pot += action.amount;
                *street_contrib.entry(action.player_id).or_insert(0.0) += action.amount;
            }

            let allin_marker = if action.is_allin { " [ALL-IN]" } else { "" };

            // ShowsCards: display the full hand — hole cards plus the board —
            // and the computed hand rank.
            if action.action == Action::ShowsCards
                && let Some(hole_cards) = &action.cards
            {
                let hole_str = hole_cards
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                let board_str = board
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                let mut eval_cards: Vec<Card> = board.clone();
                eval_cards.extend(hole_cards.iter().copied());
                let rank_str = if eval_cards.len() >= 5 {
                    let rank = eval_cards.rank();
                    format!(" ({})", CoreRank::from(rank))
                } else {
                    String::new()
                };
                let mut spans = vec![
                    Span::styled(
                        format!("  {}", player_name),
                        Style::default().fg(theme::agent_color(player_idx)),
                    ),
                    Span::styled(" shows ", Style::default().fg(TEXT)),
                    Span::styled(hole_str, Style::default().fg(YELLOW)),
                ];
                if !board_str.is_empty() {
                    spans.push(Span::styled(" | ", Style::default().fg(SURFACE2)));
                    spans.push(Span::styled(board_str, Style::default().fg(SUBTEXT0)));
                }
                spans.push(Span::styled(rank_str, Style::default().fg(MAUVE)));
                lines.push(Line::from(spans));
                continue;
            }

            let action_name = format_action(&action.action);

            if action.amount > 0.0 {
                // For a (non all-in) call, also report the total bet being
                // matched — the caller's cumulative wager this street — so it
                // reads "calls 4 to 6", not just the 4 chips added. All-in
                // calls are skipped: a short all-in doesn't match the full bet,
                // and the level it's calling isn't settled until everyone acts.
                let call_total = if action.action == Action::Call && !action.is_allin {
                    street_contrib.get(&action.player_id).copied()
                } else {
                    None
                };
                let action_text = match call_total {
                    Some(total) => format!(
                        " {} {:.0} to {:.0}{}  ",
                        action_name, action.amount, total, allin_marker
                    ),
                    None => format!(" {} {:.0}{}  ", action_name, action.amount, allin_marker),
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {}", player_name),
                        Style::default().fg(theme::agent_color(player_idx)),
                    ),
                    Span::styled(action_text, action_style),
                    Span::styled(
                        format!("(pot: {:.0})", running_pot),
                        Style::default().fg(SURFACE2),
                    ),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {}", player_name),
                        Style::default().fg(theme::agent_color(player_idx)),
                    ),
                    Span::styled(format!(" {}{}", action_name, allin_marker), action_style),
                ]));
            }
        }
        lines.push(Line::default());
    }

    // Awards section. Group wins under each pot so side pots are explicit:
    // the first pot with awards is the main pot, the rest are side pots.
    let awarded_pots: Vec<&_> = hand
        .pots
        .iter()
        .filter(|pot| !pot.player_wins.is_empty())
        .collect();
    if !awarded_pots.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            format!("── {} Awards ──", ICON_AWARD),
            Style::default().fg(LAVENDER).add_modifier(Modifier::BOLD),
        )]));
        for (pot_idx, pot) in awarded_pots.iter().enumerate() {
            let pot_label = if pot_idx == 0 {
                "Main pot".to_string()
            } else {
                format!("Side pot {}", pot_idx)
            };
            lines.push(Line::from(vec![
                Span::styled(format!("  {}", pot_label), Style::default().fg(YELLOW)),
                Span::styled(
                    format!(" ({:.0})", pot.amount),
                    Style::default().fg(SURFACE2),
                ),
            ]));
            for win in &pot.player_wins {
                let player_name = player_names.get(&win.player_id).copied().unwrap_or("?");
                let player_idx = player_index.get(&win.player_id).copied().unwrap_or(0);
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("    {}", player_name),
                        Style::default().fg(theme::agent_color(player_idx)),
                    ),
                    Span::styled(
                        format!(" wins {:.0}", win.win_amount),
                        profit_style(win.win_amount),
                    ),
                ]));
            }
        }
    }

    let paragraph = Paragraph::new(lines).block(block).scroll((scroll, 0));
    frame.render_widget(paragraph, area);
}

fn format_game_type(gt: &GameType) -> &'static str {
    match gt {
        GameType::Holdem => "Hold'em",
        GameType::Omaha => "Omaha",
        GameType::OmahaHiLo => "Omaha Hi/Lo",
        GameType::Stud => "Stud",
        GameType::StudHiLo => "Stud Hi/Lo",
        GameType::Draw => "Draw",
    }
}

fn format_bet_type(bt: &BetType) -> &'static str {
    match bt {
        BetType::NoLimit => "No Limit",
        BetType::PotLimit => "Pot Limit",
        BetType::FixedLimit => "Fixed Limit",
    }
}

fn format_action(action: &Action) -> &'static str {
    match action {
        Action::Fold => "folds",
        Action::Check => "checks",
        Action::Bet => "bets",
        Action::Raise => "raises to",
        Action::Call => "calls",
        Action::PostSmallBlind => "posts SB",
        Action::PostBigBlind => "posts BB",
        Action::PostAnte => "posts ante",
        Action::Straddle => "straddles",
        Action::PostDead => "posts dead",
        Action::PostExtraBlind => "posts extra blind",
        Action::ShowsCards => "shows",
        Action::MucksCards => "mucks",
        Action::AddedToPot => "added to pot",
        Action::DealtCards | Action::SitsDown | Action::StandsUp | Action::AddedChips => "",
    }
}

fn render_detail_keybindings(frame: &mut Frame, area: Rect) {
    let line = Line::from(vec![
        Span::styled("q", keybinding_key_style()),
        Span::styled(" Back  ", Style::default().fg(OVERLAY0)),
        Span::styled("j", keybinding_key_style()),
        Span::styled("/", Style::default().fg(OVERLAY0)),
        Span::styled("k", keybinding_key_style()),
        Span::styled(" Scroll  ", Style::default().fg(OVERLAY0)),
        Span::styled("^d", keybinding_key_style()),
        Span::styled("/", Style::default().fg(OVERLAY0)),
        Span::styled("^u", keybinding_key_style()),
        Span::styled(" ½Page  ", Style::default().fg(OVERLAY0)),
        Span::styled("g", keybinding_key_style()),
        Span::styled(" Top  ", Style::default().fg(OVERLAY0)),
        Span::styled("G", keybinding_key_style()),
        Span::styled(" End", Style::default().fg(OVERLAY0)),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

/// Count the total lines in the round log content (for scroll-to-end calculation).
pub fn round_log_line_count(hand: &HandHistory) -> u16 {
    let mut count: usize = 0;
    for round in &hand.rounds {
        count += 1; // street header
        for action in &round.actions {
            if !matches!(
                action.action,
                Action::DealtCards | Action::SitsDown | Action::StandsUp | Action::AddedChips
            ) {
                count += 1;
            }
        }
        count += 1; // blank line
    }
    // Awards section: header + one line per awarded pot + one line per win.
    let awarded_pots: Vec<&_> = hand
        .pots
        .iter()
        .filter(|pot| !pot.player_wins.is_empty())
        .collect();
    if !awarded_pots.is_empty() {
        count += 1; // awards header
        for pot in &awarded_pots {
            count += 1; // pot label
            count += pot.player_wins.len();
        }
    }
    count as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use ratatui::{Terminal, backend::TestBackend};
    use rs_poker::core::{Suit, Value};
    use rs_poker::open_hand_history::*;

    fn card(value: Value, suit: Suit) -> Card {
        Card::new(value, suit)
    }

    fn make_test_hand() -> HandHistory {
        HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "PokerStars".into(),
            network_name: "PokerStars".into(),
            internal_version: "1.0".into(),
            tournament: false,
            tournament_info: None,
            game_number: "12345".into(),
            start_date_utc: None,
            table_name: "Main".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: Some(BetLimitObj {
                bet_type: BetType::NoLimit,
                bet_cap: 0.0,
            }),
            table_size: 6,
            currency: "USD".into(),
            dealer_seat: 1,
            small_blind_amount: 1.0,
            big_blind_amount: 2.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players: vec![
                PlayerObj {
                    id: 1,
                    seat: 1,
                    name: "Alice".into(),
                    display: None,
                    starting_stack: 100.0,
                    player_bounty: None,
                    is_sitting_out: None,
                },
                PlayerObj {
                    id: 2,
                    seat: 2,
                    name: "Bob".into(),
                    display: None,
                    starting_stack: 100.0,
                    player_bounty: None,
                    is_sitting_out: None,
                },
                PlayerObj {
                    id: 3,
                    seat: 3,
                    name: "Charlie".into(),
                    display: None,
                    starting_stack: 100.0,
                    player_bounty: None,
                    is_sitting_out: None,
                },
            ],
            rounds: vec![
                RoundObj {
                    id: 0,
                    street: "Preflop".into(),
                    cards: None,
                    actions: vec![
                        ActionObj {
                            action_number: 0,
                            player_id: 1,
                            action: Action::DealtCards,
                            amount: 0.0,
                            is_allin: false,
                            cards: Some(vec![
                                card(Value::Ace, Suit::Spade),
                                card(Value::King, Suit::Spade),
                            ]),
                        },
                        ActionObj {
                            action_number: 1,
                            player_id: 2,
                            action: Action::DealtCards,
                            amount: 0.0,
                            is_allin: false,
                            cards: None,
                        },
                        ActionObj {
                            action_number: 2,
                            player_id: 1,
                            action: Action::PostSmallBlind,
                            amount: 1.0,
                            is_allin: false,
                            cards: None,
                        },
                        ActionObj {
                            action_number: 3,
                            player_id: 2,
                            action: Action::PostBigBlind,
                            amount: 2.0,
                            is_allin: false,
                            cards: None,
                        },
                        ActionObj {
                            action_number: 4,
                            player_id: 3,
                            action: Action::Fold,
                            amount: 0.0,
                            is_allin: false,
                            cards: None,
                        },
                        ActionObj {
                            action_number: 5,
                            player_id: 1,
                            action: Action::Raise,
                            amount: 6.0,
                            is_allin: false,
                            cards: None,
                        },
                        ActionObj {
                            action_number: 6,
                            player_id: 2,
                            action: Action::Call,
                            amount: 4.0,
                            is_allin: false,
                            cards: None,
                        },
                    ],
                },
                RoundObj {
                    id: 1,
                    street: "Flop".into(),
                    cards: Some(vec![
                        card(Value::Ten, Suit::Heart),
                        card(Value::Jack, Suit::Heart),
                        card(Value::Queen, Suit::Heart),
                    ]),
                    actions: vec![
                        ActionObj {
                            action_number: 7,
                            player_id: 1,
                            action: Action::Bet,
                            amount: 8.0,
                            is_allin: false,
                            cards: None,
                        },
                        ActionObj {
                            action_number: 8,
                            player_id: 2,
                            action: Action::Fold,
                            amount: 0.0,
                            is_allin: false,
                            cards: None,
                        },
                    ],
                },
            ],
            pots: vec![PotObj {
                number: 0,
                amount: 20.0,
                rake: None,
                jackpot: None,
                player_wins: vec![PlayerWinsObj {
                    player_id: 1,
                    win_amount: 20.0,
                    cashout_amount: None,
                    contributed_rake: None,
                    cashout_fee: None,
                    bonus_amount: None,
                }],
            }],
            tournament_bounties: None,
        }
    }

    /// A three-way hand that ends in a side pot at showdown.
    ///
    /// Alice (50) is short and goes all-in; Bob (200) and Charlie (200) keep
    /// betting, building a side pot only they can win. Alice flops the nuts
    /// (royal flush) and wins the main pot; Bob's trip queens beat Charlie's
    /// trip deuces for the side pot — a clean illustration that the best hand
    /// only wins the pots it was eligible for.
    fn make_side_pot_hand() -> HandHistory {
        let dealt = |player_id: u64, c0: Card, c1: Card| ActionObj {
            action_number: 0,
            player_id,
            action: Action::DealtCards,
            amount: 0.0,
            is_allin: false,
            cards: Some(vec![c0, c1]),
        };
        let shows = |player_id: u64, c0: Card, c1: Card| ActionObj {
            action_number: 0,
            player_id,
            action: Action::ShowsCards,
            amount: 0.0,
            is_allin: false,
            cards: Some(vec![c0, c1]),
        };
        let bet = |player_id: u64, action: Action, amount: f32, is_allin: bool| ActionObj {
            action_number: 0,
            player_id,
            action,
            amount,
            is_allin,
            cards: None,
        };

        HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "PokerStars".into(),
            network_name: "PokerStars".into(),
            internal_version: "1.0".into(),
            tournament: false,
            tournament_info: None,
            game_number: "67890".into(),
            start_date_utc: None,
            table_name: "Main".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: Some(BetLimitObj {
                bet_type: BetType::NoLimit,
                bet_cap: 0.0,
            }),
            table_size: 6,
            currency: "USD".into(),
            dealer_seat: 3,
            small_blind_amount: 1.0,
            big_blind_amount: 2.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players: vec![
                PlayerObj {
                    id: 1,
                    seat: 1,
                    name: "Alice".into(),
                    display: None,
                    starting_stack: 50.0,
                    player_bounty: None,
                    is_sitting_out: None,
                },
                PlayerObj {
                    id: 2,
                    seat: 2,
                    name: "Bob".into(),
                    display: None,
                    starting_stack: 200.0,
                    player_bounty: None,
                    is_sitting_out: None,
                },
                PlayerObj {
                    id: 3,
                    seat: 3,
                    name: "Charlie".into(),
                    display: None,
                    starting_stack: 200.0,
                    player_bounty: None,
                    is_sitting_out: None,
                },
            ],
            rounds: vec![
                RoundObj {
                    id: 0,
                    street: "Preflop".into(),
                    cards: None,
                    actions: vec![
                        dealt(
                            1,
                            card(Value::Ace, Suit::Heart),
                            card(Value::King, Suit::Heart),
                        ),
                        dealt(
                            2,
                            card(Value::Queen, Suit::Spade),
                            card(Value::Queen, Suit::Diamond),
                        ),
                        dealt(
                            3,
                            card(Value::Two, Suit::Spade),
                            card(Value::Two, Suit::Diamond),
                        ),
                        bet(1, Action::PostSmallBlind, 1.0, false),
                        bet(2, Action::PostBigBlind, 2.0, false),
                        bet(3, Action::Raise, 6.0, false),
                        bet(1, Action::Call, 5.0, false), // calls 5 to 6
                        bet(2, Action::Call, 4.0, false), // calls 4 to 6
                    ],
                },
                RoundObj {
                    id: 1,
                    street: "Flop".into(),
                    cards: Some(vec![
                        card(Value::Ten, Suit::Heart),
                        card(Value::Jack, Suit::Heart),
                        card(Value::Queen, Suit::Heart),
                    ]),
                    actions: vec![
                        bet(1, Action::Bet, 44.0, true), // all-in, short stack
                        bet(2, Action::Raise, 100.0, false),
                        bet(3, Action::Call, 100.0, false), // calls 100 to 100
                    ],
                },
                RoundObj {
                    id: 2,
                    street: "Turn".into(),
                    cards: Some(vec![card(Value::Two, Suit::Club)]),
                    actions: vec![],
                },
                RoundObj {
                    id: 3,
                    street: "River".into(),
                    cards: Some(vec![card(Value::Five, Suit::Diamond)]),
                    actions: vec![
                        shows(
                            1,
                            card(Value::Ace, Suit::Heart),
                            card(Value::King, Suit::Heart),
                        ),
                        shows(
                            2,
                            card(Value::Queen, Suit::Spade),
                            card(Value::Queen, Suit::Diamond),
                        ),
                        shows(
                            3,
                            card(Value::Two, Suit::Spade),
                            card(Value::Two, Suit::Diamond),
                        ),
                    ],
                },
            ],
            pots: vec![
                PotObj {
                    number: 1,
                    amount: 150.0,
                    rake: None,
                    jackpot: None,
                    player_wins: vec![PlayerWinsObj {
                        player_id: 1,
                        win_amount: 150.0,
                        cashout_amount: None,
                        contributed_rake: None,
                        cashout_fee: None,
                        bonus_amount: None,
                    }],
                },
                PotObj {
                    number: 2,
                    amount: 112.0,
                    rake: None,
                    jackpot: None,
                    player_wins: vec![PlayerWinsObj {
                        player_id: 2,
                        win_amount: 112.0,
                        cashout_amount: None,
                        contributed_rake: None,
                        cashout_fee: None,
                        bonus_amount: None,
                    }],
                },
            ],
            tournament_bounties: None,
        }
    }

    #[test]
    fn test_render_detail_side_pots() {
        let backend = TestBackend::new(80, 44);
        let mut terminal = Terminal::new(backend).unwrap();
        let hand = make_side_pot_hand();
        terminal
            .draw(|frame| {
                render_detail(frame, &hand, 0);
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }

    #[test]
    fn test_side_pot_profits_and_line_count() {
        let hand = make_side_pot_hand();
        let (id_to_idx, profits) = compute_hand_profits(&hand);
        // Alice: invested 1+5+44 = 50, won 150 → +100
        assert!((profits[id_to_idx[&1]] - 100.0).abs() < 0.01);
        // Bob: invested 2+4+100 = 106, won 112 → +6
        assert!((profits[id_to_idx[&2]] - 6.0).abs() < 0.01);
        // Charlie: invested 6+100 = 106, won 0 → -106
        assert!((profits[id_to_idx[&3]] - (-106.0)).abs() < 0.01);

        // Preflop: 1 header + 5 visible + 1 blank = 7
        // Flop:    1 header + 3 actions + 1 blank = 5
        // Turn:    1 header + 0 actions + 1 blank = 2
        // River:   1 header + 3 shows  + 1 blank = 5
        // Awards:  1 header + 2 pot labels + 2 wins = 5
        // Total = 24
        assert_eq!(round_log_line_count(&hand), 24);
    }

    #[test]
    fn test_render_detail_full() {
        let backend = TestBackend::new(80, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        let hand = make_test_hand();
        terminal
            .draw(|frame| {
                render_detail(frame, &hand, 0);
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }

    #[test]
    fn test_render_detail_scrolled() {
        let backend = TestBackend::new(80, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        let hand = make_test_hand();
        terminal
            .draw(|frame| {
                render_detail(frame, &hand, 5);
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }

    #[test]
    fn test_render_detail_empty_hand() {
        let backend = TestBackend::new(80, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        let hand = HandHistory {
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
        };
        terminal
            .draw(|frame| {
                render_detail(frame, &hand, 0);
            })
            .unwrap();
        assert_snapshot!(terminal.backend());
    }

    #[test]
    fn test_compute_hand_profits_basic() {
        let hand = make_test_hand();
        let (id_to_idx, profits) = compute_hand_profits(&hand);
        // Alice (seat 0): posted SB 1 + raised 6 + bet 8 = 15 spent, won 20 → profit 5
        assert!((profits[id_to_idx[&1]] - 5.0).abs() < 0.01);
        // Bob (seat 1): posted BB 2 + called 4 = 6 spent, won 0 → profit -6
        assert!((profits[id_to_idx[&2]] - (-6.0)).abs() < 0.01);
        // Charlie (seat 2): no action with amount → 0 spent, 0 won → profit 0
        assert!((profits[id_to_idx[&3]] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_round_log_line_count() {
        let hand = make_test_hand();
        let count = round_log_line_count(&hand);
        // Preflop: 1 header + 5 visible actions (2 DealtCards skipped) + 1 blank = 7
        // Flop: 1 header + 2 actions + 1 blank = 4
        // Awards: 1 header + 1 pot label + 1 win = 3
        // Total = 14
        assert_eq!(count, 14);
    }

    #[test]
    fn test_round_log_line_count_no_awards() {
        let mut hand = make_test_hand();
        hand.pots.clear();
        let count = round_log_line_count(&hand);
        // Same as above minus awards section (2 lines)
        assert_eq!(count, 11);
    }

    #[test]
    fn test_format_game_type_all_variants() {
        assert_eq!(format_game_type(&GameType::Holdem), "Hold'em");
        assert_eq!(format_game_type(&GameType::Omaha), "Omaha");
        assert_eq!(format_game_type(&GameType::OmahaHiLo), "Omaha Hi/Lo");
        assert_eq!(format_game_type(&GameType::Stud), "Stud");
        assert_eq!(format_game_type(&GameType::StudHiLo), "Stud Hi/Lo");
        assert_eq!(format_game_type(&GameType::Draw), "Draw");
    }

    #[test]
    fn test_format_bet_type_all_variants() {
        assert_eq!(format_bet_type(&BetType::NoLimit), "No Limit");
        assert_eq!(format_bet_type(&BetType::PotLimit), "Pot Limit");
        assert_eq!(format_bet_type(&BetType::FixedLimit), "Fixed Limit");
    }

    #[test]
    fn test_format_action_all_variants() {
        assert_eq!(format_action(&Action::Fold), "folds");
        assert_eq!(format_action(&Action::Check), "checks");
        assert_eq!(format_action(&Action::Bet), "bets");
        assert_eq!(format_action(&Action::Raise), "raises to");
        assert_eq!(format_action(&Action::Call), "calls");
        assert_eq!(format_action(&Action::PostSmallBlind), "posts SB");
        assert_eq!(format_action(&Action::PostBigBlind), "posts BB");
        assert_eq!(format_action(&Action::PostAnte), "posts ante");
        assert_eq!(format_action(&Action::Straddle), "straddles");
        assert_eq!(format_action(&Action::PostDead), "posts dead");
        assert_eq!(format_action(&Action::PostExtraBlind), "posts extra blind");
        assert_eq!(format_action(&Action::ShowsCards), "shows");
        assert_eq!(format_action(&Action::MucksCards), "mucks");
        assert_eq!(format_action(&Action::AddedToPot), "added to pot");
        assert_eq!(format_action(&Action::DealtCards), "");
        assert_eq!(format_action(&Action::SitsDown), "");
        assert_eq!(format_action(&Action::StandsUp), "");
        assert_eq!(format_action(&Action::AddedChips), "");
    }
}
