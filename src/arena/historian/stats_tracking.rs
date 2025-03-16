use std::{cell::RefCell, rc::Rc};

use super::Historian;

use crate::arena::GameState;
use crate::arena::action::{Action, AgentAction, PlayedActionPayload};
use crate::core::Rankable;

/// Storage for tracking various poker player statistics
///
/// # Fields
///
/// * `actions_count` - Vector storing the count of total actions performed by
///   each player
/// * `vpip_count` - Vector storing the count of voluntary put in pot (VPIP)
///   actions for each player
/// * `vpip_total` - Vector storing the running total of VPIP percentage for
///   each player
/// * `vpip_ahead_count` - Vector storing the count of VPIP actions when ahead
///   in hand for each player
/// * `vpip_ahead_total` - Vector storing the running total of VPIP percentage
///   when ahead for each player
/// * `raise_count` - Vector storing the count of raise actions performed by
///   each player
/// * `raise_ahead_count` - Vector storing the count of raise actions when ahead
///   in hand for each player
pub struct StatsStorage {
    // The total number of actions each player has taken
    pub actions_count: Vec<usize>,
    // How many times each player has voluntarily put money in the pot
    pub vpip_count: Vec<usize>,
    // The total amount of money each player has voluntarily put in the pot
    pub vpip_total: Vec<f32>,

    // How many times they were ahead
    pub vpip_ahead_count: Vec<usize>,
    // They ammount they were ahead and bet
    pub vpip_ahead_total: Vec<f32>,

    // How many times they raised
    pub raise_count: Vec<usize>,

    pub raise_ahead_count: Vec<usize>,
}

impl StatsStorage {
    pub fn new_with_num_players(num_players: usize) -> Self {
        Self {
            actions_count: vec![0; num_players],

            vpip_count: vec![0; num_players],
            vpip_total: vec![0.0; num_players],

            vpip_ahead_count: vec![0; num_players],
            vpip_ahead_total: vec![0.0; num_players],

            raise_count: vec![0; num_players],

            raise_ahead_count: vec![0; num_players],
        }
    }
}

impl Default for StatsStorage {
    fn default() -> Self {
        StatsStorage::new_with_num_players(9)
    }
}

/// A historian implementation that tracks and stores poker game statistics
///
/// # Fields
/// * `storage` - A reference-counted, mutable reference to the statistics
///   storage
pub struct StatsTrackingHistorian {
    storage: Rc<RefCell<StatsStorage>>,
}

impl StatsTrackingHistorian {
    pub fn get_storage(&self) -> Rc<RefCell<StatsStorage>> {
        self.storage.clone()
    }

    fn record_played_action(
        &mut self,
        games_state: &GameState,
        payload: PlayedActionPayload,
    ) -> Result<(), super::HistorianError> {
        let ranks = games_state
            .hands
            .iter()
            .map(|hand| hand.rank())
            .collect::<Vec<_>>();

        let max_hand = ranks.iter().max().unwrap();
        let is_behind = ranks.get(payload.idx).unwrap() < max_hand;

        let mut storage = self.storage.try_borrow_mut()?;
        storage.actions_count[payload.idx] += 1;

        if let AgentAction::Bet(bet_ammount) = payload.action {
            let put_into_pot = bet_ammount - payload.starting_player_bet;
            if put_into_pot > 0.0 {
                // Played Action Payloads can't come from a forced bet
                // so if there's a bet amount, it's a voluntary action
                storage.vpip_count[payload.idx] += 1;
                // They put in the bet ammount minus what they already had in the pot
                storage.vpip_total[payload.idx] += put_into_pot;

                // If they were ahead then track that
                if !is_behind {
                    storage.vpip_ahead_count[payload.idx] += 1;
                    storage.vpip_ahead_total[payload.idx] += put_into_pot;
                }
            }

            // they raised
            if payload.starting_bet > payload.final_bet {
                storage.raise_count[payload.idx] += 1;
                if !is_behind {
                    storage.raise_ahead_count[payload.idx] += 1;
                }
            }
        }

        Ok(())
    }

    pub fn new_with_num_players(num_players: usize) -> Self {
        Self {
            storage: Rc::new(RefCell::new(StatsStorage::new_with_num_players(
                num_players,
            ))),
        }
    }
}

impl Default for StatsTrackingHistorian {
    fn default() -> Self {
        Self {
            storage: Rc::new(RefCell::new(StatsStorage::default())),
        }
    }
}

impl Historian for StatsTrackingHistorian {
    fn record_action(
        &mut self,
        _id: &uuid::Uuid,
        game_state: &GameState,
        action: Action,
    ) -> Result<(), super::HistorianError> {
        match action {
            Action::PlayedAction(payload) => self.record_played_action(game_state, payload),
            Action::FailedAction(failed_action_payload) => {
                self.record_played_action(game_state, failed_action_payload.result)
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::{agent::{AllInAgent, CallingAgent, FoldingAgent}, Agent, HoldemSimulationBuilder};

    use super::*;

    #[test]
    fn test_all_in_agents_had_actions_counted() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<AllInAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run();

		println!("{:?}", storage
			.borrow()
			.actions_count);

        assert!(
            storage
                .borrow()
                .actions_count
                .iter()
                .all(|&count| count == 1)
        );
    }

    #[test]
    fn test_calling_agents_had_actions_counted() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<CallingAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run();

		println!("{:?}", storage
			.borrow()
			.actions_count);

        assert!(
            storage
                .borrow()
                .actions_count
                .iter()
                .all(|&count| count == 4)
        );
    }

    #[test]
    fn test_folding_agents_had_actions_counted() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run();

		println!("{:?}", storage
			.borrow()
			.actions_count);

		let actions_count = &storage.borrow().actions_count;

		// Player 0 folded before player 1 could even act.
        assert_eq!(actions_count.get(0), Some(&1));
        assert_eq!(actions_count.get(1), Some(&0));
    }
}
