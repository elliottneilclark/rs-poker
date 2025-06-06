use serde::{Deserialize, Serialize};

use crate::core::Card;
use crate::open_hand_history::serde_utils::{empty_string_is_none, iso8601};
use chrono::{DateTime, Utc};

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpeedType {
    Normal,
    #[serde(rename = "Semi-Turbo")]
    SemiTurbo,
    #[serde(rename = "Turbo")]
    Turbo,
    #[serde(rename = "Super-Turbo")]
    SuperTurbo,
    #[serde(rename = "Hyper-Turbo")]
    HyperTurbo,
    #[serde(rename = "Ultra-Turbo")]
    UltraTurbo,
}

/// Defines the speed/timing history of a tournament
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeedObj {
    /// Tournament speed category
    ///
    /// Defines how quickly blind levels increase relative to normal
    /// tournaments. See SpeedType enum for possible values like Normal,
    /// Turbo, etc.
    #[serde(rename = "type")]
    pub speed_type: SpeedType,

    /// Round time in seconds
    ///
    /// The amount of time (in seconds) between blinds increases. For example,
    /// if blinds go up every 10 minutes, this would be 600 seconds.
    pub round_time: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TournamentBountyObj {
    pub player_id: u64,
    pub bounty_won: f32,
    pub defeated_player_id: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TournamentType {
    #[serde(rename = "STT")]
    SingleTableTournament,
    #[serde(rename = "MTT")]
    MultiTableTournament,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TournamentFlag {
    /// Sit 'N Go tournament
    #[serde(rename = "SNG")]
    SitNGo,
    /// Double or nothing - Half of players win double buy-in
    #[serde(rename = "DON")]
    DoubleOrNothing,
    /// Players are awarded prizes for knocking out specific players
    Bounty,
    /// Multi-table tournament played in stages where each table plays until one
    /// remains
    Shootout,
    /// Players can rebuy stacks of chips during specified periods
    Rebuy,
    /// One buyin split between multiple games with aggregate prizes
    Matrix,
    /// Players must either fold or push all-in every hand
    #[serde(rename = "Push_Or_Fold")]
    PushOrFold,
    /// Prize includes entry into another tournament
    Satellite,
    /// Incremental tournament progression to higher value tournaments
    Steps,
    /// Starting stacks are deeper than normal tournaments
    Deep,
    #[serde(rename = "Multi-Entry")]
    /// Players can have multiple entries at tournament start
    MultiEntry,
    /// 50% of players win equal base prize plus chip-based bonus
    #[serde(rename = "Fifty50")]
    FiftyFifty,
    /// All hands played face up without betting rounds
    Flipout,
    /// One third of players win triple their buy-in
    TripleUp,
    /// Fast games with random prizes determined before start
    Lottery,
    #[serde(rename = "Re-Entry")]
    /// Players can re-enter after elimination
    ReEntry,
    #[serde(rename = "Power_Up")]
    /// Three person game with special powers
    PowerUp,
    #[serde(rename = "Progressive-Bounty")]
    /// Bounty tournaments where bounty amounts can vary and accumulate
    ProgressiveBounty,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TournamentInfoObj {
    pub tournament_number: String,
    pub name: String,
    #[serde(with = "iso8601")]
    pub start_date_utc: Option<DateTime<Utc>>,
    pub currency: String,
    pub buyin_amount: f32,
    pub fee_amount: f32,
    pub bounty_fee_amount: f32,
    pub initial_stack: u64,
    #[serde(rename = "type")]
    pub tournament_type: TournamentType,
    #[serde(deserialize_with = "empty_string_is_none")]
    pub flags: Option<Vec<TournamentFlag>>,
    pub speed: SpeedObj,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlayerWinsObj {
    pub player_id: u64,
    pub win_amount: f32,
    pub cashout_amount: Option<f32>,
    pub cashout_fee: Option<f32>,
    pub bonus_amount: Option<f32>,
    pub contributed_rake: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PotObj {
    pub number: u64,
    pub amount: f32,
    pub rake: Option<f32>,
    pub jackpot: Option<f32>,
    pub player_wins: Vec<PlayerWinsObj>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    /// Player is dealt cards
    #[serde(rename = "Dealt Cards")]
    DealtCards,
    /// Player mucks (discards) their cards
    #[serde(rename = "Mucks Cards")]
    MucksCards,
    /// Player shows their cards
    #[serde(rename = "Shows Cards")]
    ShowsCards,
    /// Player posts the ante
    #[serde(rename = "Post Ante")]
    PostAnte,
    /// Player posts the small blind
    #[serde(rename = "Post SB")]
    PostSmallBlind,
    /// Player posts the big blind
    #[serde(rename = "Post BB")]
    PostBigBlind,
    /// Player posts a straddle bet
    #[serde(rename = "Straddle")]
    Straddle,
    /// Player posts a dead blind
    #[serde(rename = "Post Dead")]
    PostDead,
    /// Player posts an extra blind
    #[serde(rename = "Post Extra Blind")]
    PostExtraBlind,
    /// Player folds their hand
    #[serde(rename = "Fold")]
    Fold,
    /// Player checks
    #[serde(rename = "Check")]
    Check,
    /// Player makes a bet
    #[serde(rename = "Bet")]
    Bet,
    /// Player raises a previous bet
    #[serde(rename = "Raise")]
    Raise,
    /// Player calls a previous bet
    #[serde(rename = "Call")]
    Call,
    /// Player adds chips to their stack
    #[serde(rename = "Added Chips")]
    AddedChips,
    /// Player sits down at the table
    #[serde(rename = "Sits Down")]
    SitsDown,
    /// Player stands up from the table
    #[serde(rename = "Stands Up")]
    StandsUp,
    /// Chips added to pot (usually from dead blinds/antes)
    #[serde(rename = "Added To Pot")]
    AddedToPot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionObj {
    pub action_number: u64,
    pub player_id: u64,
    pub action: Action,
    pub amount: f32,
    pub is_allin: bool,
    #[serde(default, deserialize_with = "empty_string_is_none")]
    pub cards: Option<Vec<Card>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundObj {
    pub id: u64,
    pub street: String,
    #[serde(default, deserialize_with = "empty_string_is_none")]
    pub cards: Option<Vec<Card>>,
    pub actions: Vec<ActionObj>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlayerObj {
    pub id: u64,
    pub seat: u64,
    pub name: String,
    pub display: Option<String>,
    pub starting_stack: f32,
    pub player_bounty: Option<f32>,
    pub is_sitting_out: Option<bool>,
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BetType {
    #[serde(rename = "NL")]
    NoLimit,
    #[serde(rename = "PL")]
    PotLimit,
    #[serde(rename = "FL")]
    FixedLimit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BetLimitObj {
    pub bet_type: BetType,
    pub bet_cap: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GameType {
    Holdem,
    Omaha,
    OmahaHiLo,
    Stud,
    StudHiLo,
    Draw,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HandFlag {
    /// Two or more sets of board cards are used in the play of the same hand
    #[serde(rename = "Run_It_Twice")]
    RunItTwice,
    /// Players at the table are anonymous without specified names/IDs
    Anonymous,
    /// Hand was observed without hero being dealt in
    Observed,
    /// Fast poker variant where folding moves to next hand immediately
    Fast,
    /// Game limits the total amount each player can wager per hand
    Cap,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandHistory {
    // Current version is 1.4.7
    pub spec_version: String,
    pub site_name: String,
    pub network_name: String,
    pub internal_version: String,
    #[serde(default)]
    pub tournament: bool,
    pub tournament_info: Option<TournamentInfoObj>,
    pub game_number: String,
    #[serde(with = "iso8601")]
    pub start_date_utc: Option<DateTime<Utc>>,
    pub table_name: String,
    pub table_handle: Option<String>,
    pub table_skin: Option<String>,
    pub game_type: GameType,
    pub bet_limit: Option<BetLimitObj>,
    pub table_size: u64,
    pub currency: String,
    pub dealer_seat: u64,
    pub small_blind_amount: f32,
    pub big_blind_amount: f32,
    pub ante_amount: f32,

    // Which player is the hero and being followed
    pub hero_player_id: Option<u64>,
    // #[serde(deserialize_with = "empty_string_is_none")]
    // pub flags: Option<Vec<HandFlag>>,
    pub players: Vec<PlayerObj>,
    pub rounds: Vec<RoundObj>,
    pub pots: Vec<PotObj>,
    pub tournament_bounties: Option<Vec<TournamentBountyObj>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenHandHistoryWrapper {
    pub ohh: HandHistory,
}

#[cfg(test)]
mod tests {
    use super::{OpenHandHistoryWrapper, TournamentType};

    #[test]
    fn test_ipoker_example() {
        let json = r#"
                    {
                        "ohh": {
                            "spec_version": "1.2.2",
                            "internal_version": "1.2.2",
                            "network_name": "iPoker Network",
                            "site_name": "iPoker",
                            "game_type": "Holdem",
                            "table_name": "Brehuiesti",
                            "table_size": 6,
                            "game_number": "8599887794",
                            "start_date_utc": "2020-04-07T14:32:50",
                            "currency": "EUR",
                            "ante_amount": 0.00,
                            "small_blind_amount": 0.01,
                            "big_blind_amount": 0.02,
                            "bet_limit": {
                                "bet_cap": 0.00,
                                "bet_type": "NL"
                            },
                            "hero_player_id": 3,
                            "dealer_seat": 8,
                            "players": [
                                {
                                    "name": "Player1",
                                    "id": 0,
                                    "player_bounty": 0,
                                    "starting_stack": 0.80,
                                    "seat": 1
                                },
                                {
                                    "name": "Player3",
                                    "id": 1,
                                    "player_bounty": 0,
                                    "starting_stack": 2.38,
                                    "seat": 3
                                },
                                {
                                    "name": "Player5",
                                    "id": 2,
                                    "player_bounty": 0,
                                    "starting_stack": 2.67,
                                    "seat": 5
                                },
                                {
                                    "name": "Hero",
                                    "id": 3,
                                    "player_bounty": 0,
                                    "starting_stack": 2.00,
                                    "seat": 6
                                },
                                {
                                    "name": "Player8",
                                    "id": 4,
                                    "player_bounty": 0,
                                    "starting_stack": 2.86,
                                    "seat": 8
                                },
                                {
                                    "name": "Player10",
                                    "id": 5,
                                    "player_bounty": 0,
                                    "starting_stack": 2.00,
                                    "seat": 10
                                }
                            ],
                            "rounds": [
                                {
                                    "id": 0,
                                    "cards": "",
                                    "street": "Preflop",
                                    "actions": [
                                        {
                                            "action_number": 1,
                                            "player_id": 5,
                                            "action": "Post SB",
                                            "amount": 0.01,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 2,
                                            "player_id": 0,
                                            "action": "Post BB",
                                            "amount": 0.02,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 3,
                                            "player_id": 3,
                                            "cards": [
                                                "As",
                                                "Tc"
                                            ],
                                            "action": "Dealt Cards",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 4,
                                            "player_id": 1,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 5,
                                            "player_id": 2,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 6,
                                            "player_id": 3,
                                            "action": "Raise",
                                            "amount": 0.06,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 7,
                                            "player_id": 4,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 8,
                                            "player_id": 5,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 9,
                                            "player_id": 0,
                                            "action": "Call",
                                            "amount": 0.04,
                                            "is_allin": false
                                        }
                                    ]
                                },
                                {
                                    "id": 1,
                                    "cards": [
                                        "5c",
                                        "7d",
                                        "Js"
                                    ],
                                    "street": "Flop",
                                    "actions": [
                                        {
                                            "action_number": 1,
                                            "player_id": 0,
                                            "action": "Check",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 2,
                                            "player_id": 3,
                                            "action": "Bet",
                                            "amount": 0.06,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 3,
                                            "player_id": 0,
                                            "action": "Raise",
                                            "amount": 0.18,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 4,
                                            "player_id": 3,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        }
                                    ]
                                }
                            ],
                            "pots": [
                                {
                                    "number": 0,
                                    "amount": 0.25,
                                    "rake": 0.01,
                                    "player_wins": [
                                        {
                                            "player_id": 0,
                                            "win_amount": 0.24,
                                            "contributed_rake": 0.01
                                        }
                                    ]
                                }
                            ]
                        }
                    }"#;

        let parsed: OpenHandHistoryWrapper = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.ohh.spec_version, "1.2.2");
        assert_eq!(parsed.ohh.small_blind_amount, 0.01);
        assert_eq!(parsed.ohh.big_blind_amount, 0.02);
    }

    #[test]
    fn test_poker_stars_tournament() {
        let json = r#"
                    {
                        "ohh": {
                            "spec_version": "1.2.2",
                            "internal_version": "1.2.2",
                            "network_name": "PokerStars",
                            "site_name": "PokerStars",
                            "game_type": "Holdem",
                            "start_date_utc": "2019-03-28T08:16:05",
                            "table_size": 3,
                            "table_name": "2572001822 1",
                            "game_number": "198636399064",
                            "currency": "",
                            "ante_amount": 0.00,
                            "small_blind_amount": 10.00,
                            "big_blind_amount": 20.00,
                            "bet_limit": {
                                "bet_cap": 0.00,
                                "bet_type": "NL"
                            },
                            "tournament_info": {
                                "name": "",
                                "start_date_utc": "2019-03-28T08:16:05",
                                "type": "STT",
                                "buyin_amount": 0.23,
                                "currency": "USD",
                                "fee_amount": 0.02,
                                "bounty_fee_amount": 0.00,
                                "tournament_number": "2572001822",
                                "flags": "",
                                "initial_stack": 1500,
                                "speed": {
                                    "type": "Normal",
                                    "round_time": 10
                                }
                            },
                            "tournament_bounties": [
                            ],
                            "tournament_rebuys": [
                            ],
                            "hero_player_id": 0,
                            "dealer_seat": 2,
                            "players": [
                                {
                                    "name": "Hero",
                                    "id": 0,
                                    "player_bounty": 0,
                                    "starting_stack": 500.00,
                                    "seat": 1
                                },
                                {
                                    "name": "Player2",
                                    "id": 1,
                                    "player_bounty": 0,
                                    "starting_stack": 490.00,
                                    "seat": 2
                                },
                                {
                                    "name": "Player3",
                                    "id": 2,
                                    "player_bounty": 0,
                                    "starting_stack": 510.00,
                                    "seat": 3
                                }
                            ],
                            "rounds": [
                                {
                                    "id": 0,
                                    "cards": "",
                                    "street": "Preflop",
                                    "actions": [
                                        {
                                            "action_number": 1,
                                            "player_id": 2,
                                            "action": "Post SB",
                                            "amount": 10.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 2,
                                            "player_id": 0,
                                            "action": "Post BB",
                                            "amount": 20.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 3,
                                            "player_id": 0,
                                            "cards": [
                                                "4d",
                                                "9h"
                                            ],
                                            "action": "Dealt Cards",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 4,
                                            "player_id": 1,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 5,
                                            "player_id": 2,
                                            "action": "Call",
                                            "amount": 10.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 6,
                                            "player_id": 0,
                                            "action": "Check",
                                            "amount": 0.00,
                                            "is_allin": false
                                        }
                                    ]
                                },
                                {
                                    "id": 1,
                                    "cards": [
                                        "2s",
                                        "Ac",
                                        "Js"
                                    ],
                                    "street": "Flop",
                                    "actions": [
                                        {
                                            "action_number": 1,
                                            "player_id": 2,
                                            "action": "Bet",
                                            "amount": 40.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 2,
                                            "player_id": 0,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        }
                                    ]
                                },
                                {
                                    "id": 4,
                                    "street": "Showdown",
                                    "actions": [
                                        {
                                            "action_number": 1,
                                            "player_id": 2,
                                            "action": "Mucks Cards",
                                            "amount": 0.00,
                                            "is_allin": false
                                        }
                                    ]
                                }
                            ],
                            "pots": [
                                {
                                    "rake": 0.00,
                                    "number": 0,
                                    "player_wins": [
                                        {
                                            "player_id": 2,
                                            "win_amount": 40.00,
                                            "contributed_rake": 0.00
                                        }
                                    ],
                                    "amount": 40.00
                                }
                            ]
                        }
                    }"#;

        let parsed: OpenHandHistoryWrapper = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.ohh.site_name, "PokerStars");

        assert_eq!(
            parsed.ohh.tournament_info.as_ref().unwrap().tournament_type,
            TournamentType::SingleTableTournament
        );
    }

    #[test]
    fn test_888_example_hand() {
        let json = r#"
                    {
                        "ohh": {
                            "spec_version": "1.2.2",
                            "internal_version": "1.2.2",
                            "network_name": "888 Poker",
                            "site_name": "Pacific Poker",
                            "game_type": "Holdem",
                            "table_name": "Milan 9 Max ($0.02 NL)",
                            "table_size": 9,
                            "game_number": "481325875",
                            "start_date_utc": "2017-07-14T15:03:39",
                            "currency": "USD",
                            "ante_amount": 0.00,
                            "small_blind_amount": 0.01,
                            "big_blind_amount": 0.02,
                            "bet_limit": {
                                "bet_cap": 0.00,
                                "bet_type": "NL"
                            },
                            "hero_player_id": 8,
                            "dealer_seat": 4,
                            "players": [
                                {
                                    "name": "Player1",
                                    "id": 0,
                                    "player_bounty": 0,
                                    "starting_stack": 0.57,
                                    "seat": 1
                                },
                                {
                                    "name": "Player2",
                                    "id": 1,
                                    "player_bounty": 0,
                                    "starting_stack": 2.22,
                                    "seat": 2
                                },
                                {
                                    "name": "Player3",
                                    "id": 2,
                                    "player_bounty": 0,
                                    "starting_stack": 1.95,
                                    "seat": 3
                                },
                                {
                                    "name": "Player4",
                                    "id": 3,
                                    "player_bounty": 0,
                                    "starting_stack": 2.11,
                                    "seat": 4
                                },
                                {
                                    "name": "Player5",
                                    "id": 4,
                                    "player_bounty": 0,
                                    "starting_stack": 1.87,
                                    "seat": 5
                                },
                                {
                                    "name": "Player6",
                                    "id": 5,
                                    "player_bounty": 0,
                                    "starting_stack": 2.00,
                                    "seat": 6
                                },
                                {
                                    "name": "Player7",
                                    "id": 6,
                                    "player_bounty": 0,
                                    "starting_stack": 3.02,
                                    "seat": 7
                                },
                                {
                                    "name": "Player9",
                                    "id": 7,
                                    "player_bounty": 0,
                                    "starting_stack": 2.11,
                                    "seat": 9
                                },
                                {
                                    "name": "Hero",
                                    "id": 8,
                                    "player_bounty": 0,
                                    "starting_stack": 1.88,
                                    "seat": 10
                                }
                            ],
                            "rounds": [
                                {
                                    "id": 0,
                                    "cards": "",
                                    "street": "Preflop",
                                    "actions": [
                                        {
                                            "action_number": 1,
                                            "player_id": 4,
                                            "action": "Post SB",
                                            "amount": 0.01,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 2,
                                            "player_id": 5,
                                            "action": "Post BB",
                                            "amount": 0.02,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 3,
                                            "player_id": 8,
                                            "cards": [
                                                "Ad",
                                                "Jh"
                                            ],
                                            "action": "Dealt Cards",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 4,
                                            "player_id": 6,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 5,
                                            "player_id": 7,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 6,
                                            "player_id": 8,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 7,
                                            "player_id": 0,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 8,
                                            "player_id": 1,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 9,
                                            "player_id": 2,
                                            "action": "Call",
                                            "amount": 0.02,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 10,
                                            "player_id": 3,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 11,
                                            "player_id": 4,
                                            "action": "Call",
                                            "amount": 0.01,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 12,
                                            "player_id": 5,
                                            "action": "Check",
                                            "amount": 0.00,
                                            "is_allin": false
                                        }
                                    ]
                                },
                                {
                                    "id": 1,
                                    "cards": [
                                        "8d",
                                        "Kh",
                                        "2s"
                                    ],
                                    "street": "Flop",
                                    "actions": [
                                        {
                                            "action_number": 1,
                                            "player_id": 4,
                                            "action": "Check",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 2,
                                            "player_id": 5,
                                            "action": "Check",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 3,
                                            "player_id": 2,
                                            "action": "Bet",
                                            "amount": 0.06,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 4,
                                            "player_id": 4,
                                            "action": "Raise",
                                            "amount": 0.12,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 5,
                                            "player_id": 5,
                                            "action": "Fold",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 6,
                                            "player_id": 2,
                                            "action": "Call",
                                            "amount": 0.06,
                                            "is_allin": false
                                        }
                                    ]
                                },
                                {
                                    "id": 2,
                                    "cards": [
                                        "Ts"
                                    ],
                                    "street": "Turn",
                                    "actions": [
                                        {
                                            "action_number": 1,
                                            "player_id": 4,
                                            "action": "Bet",
                                            "amount": 0.22,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 2,
                                            "player_id": 2,
                                            "action": "Call",
                                            "amount": 0.22,
                                            "is_allin": false
                                        }
                                    ]
                                },
                                {
                                    "id": 3,
                                    "cards": [
                                        "Kd"
                                    ],
                                    "street": "River",
                                    "actions": [
                                        {
                                            "action_number": 1,
                                            "player_id": 4,
                                            "action": "Bet",
                                            "amount": 0.16,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 2,
                                            "player_id": 2,
                                            "action": "Call",
                                            "amount": 0.16,
                                            "is_allin": false
                                        }
                                    ]
                                },
                                {
                                    "id": 4,
                                    "street": "Showdown",
                                    "actions": [
                                        {
                                            "action_number": 1,
                                            "player_id": 4,
                                            "cards": [
                                                "7d",
                                                "Kc"
                                            ],
                                            "action": "Shows Cards",
                                            "amount": 0.00,
                                            "is_allin": false
                                        },
                                        {
                                            "action_number": 2,
                                            "player_id": 2,
                                            "cards": [
                                                "Ks",
                                                "4s"
                                            ],
                                            "action": "Shows Cards",
                                            "amount": 0.00,
                                            "is_allin": false
                                        }
                                    ]
                                }
                            ],
                            "pots": [
                                {
                                    "number": 0,
                                    "amount": 1.06,
                                    "rake": 0.06,
                                    "player_wins": [
                                        {
                                            "player_id": 2,
                                            "win_amount": 0.50,
                                            "contributed_rake": 0.03
                                        },
                                        {
                                            "player_id": 4,
                                            "win_amount": 0.50,
                                            "contributed_rake": 0.03
                                        }
                                    ]
                                }
                            ]
                        }
                    }
        "#;

        let parsed: OpenHandHistoryWrapper = serde_json::from_str(json).unwrap();

        assert_eq!(parsed.ohh.site_name, "Pacific Poker");
        assert_eq!(parsed.ohh.network_name, "888 Poker");
    }
}
