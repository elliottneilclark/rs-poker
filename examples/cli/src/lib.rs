use std::fmt::Display;

use rs_poker::core::{Hand, RSPokerError};

use inquire::{error::InquireResult, validator::Validation, Select, Text};

#[derive(Debug, Copy, Clone)]
pub enum GameType {
    MonteCarlo,
    Omaha,
}

impl Display for GameType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Debug>::fmt(&self, f)
    }
}

pub struct Config {
    pub game_type: GameType,
    pub num_of_players: u8,
    pub hands: Vec<Hand>,
    pub num_of_games_to_simulate: u64,
}

impl Config {
    pub fn new() -> InquireResult<Self> {
        let game_type =
            Select::new("Type of game:", vec![GameType::MonteCarlo, GameType::Omaha]).prompt()?;

        let num_of_players = Text::new("Number of players:")
            .with_help_message("requires a number between 2 - 10")
            .with_validator(|input: &str| {
                const NUM_PLAYERS_ERR: &'static str = "Please enter a number between 2 to 10";

                if let Ok(parsed_num) = u8::from_str_radix(input, 10) {
                    if parsed_num <= 10 && parsed_num >= 2 {
                        Ok(Validation::Valid)
                    } else {
                        Ok(Validation::Invalid(NUM_PLAYERS_ERR.into()))
                    }
                } else {
                    Ok(Validation::Invalid(NUM_PLAYERS_ERR.into()))
                }
            })
            .prompt()?
            .parse::<u8>()
            //[SAFETY]: this should be never panic as we already did the necessary validation
            .unwrap();

        let hands = (0..num_of_players)
            .map(|player_num| -> InquireResult<Hand> {
                let hand = Text::new(&format!("Hand of player {}:", player_num + 1))
                    .with_help_message("A valid hand is contiguous string of valid cards. Each card is a 2 character sequence starting with a valid `Value` identifier, followed by a valid `Suite` identifier")
                    .with_validator(|input: &str| {
                        match input.parse::<Hand>() {
                            Ok(_) => Ok(Validation::Valid),
                            Err(RSPokerError::UnexpectedValueChar) => Ok(Validation::Invalid("Value can only be a char from [`A`, `K`, `Q`, `J`, `T`, `9`, `8`, `7`, `6`, `5`, `4`, `3`, `2`]".into())),
                            Err(RSPokerError::UnexpectedSuitChar) => Ok(Validation::Invalid("Suite can only be a char from [`d`, `s`, `h`, `c`]".into())),
                            Err(RSPokerError::DuplicateCardInHand(c)) => Ok(Validation::Invalid(format!("Card {c} appears more than once").into())),
                            Err(RSPokerError::UnparsedCharsRemaining) => Ok(Validation::Invalid("Extraneous trailing characters encountered".into())),
                            _ => unreachable!()
                        }
                    })
                    .prompt()?
                    .parse::<Hand>()
                    //[SAFETY]: this should be never panic as we already did the necessary validation
                    .unwrap();

                Ok(hand)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let num_of_games_to_simulate = Text::new("Number of games to simulate:")
            .with_validator(|input: &str| {
                const NUM_PLAYERS_ERR: &'static str = "Please enter a number greater than 0";

                if let Ok(parsed_num) = u64::from_str_radix(input, 10) {
                    if parsed_num > 0 {
                        Ok(Validation::Valid)
                    } else {
                        Ok(Validation::Invalid(NUM_PLAYERS_ERR.into()))
                    }
                } else {
                    Ok(Validation::Invalid(NUM_PLAYERS_ERR.into()))
                }
            })
            .prompt()?
            .parse::<u64>()
            //[SAFETY]: this should be never panic as we already did the necessary validation
            .unwrap();

        Ok(Self {
            game_type,
            num_of_games_to_simulate,
            num_of_players,
            hands,
        })
    }
}
