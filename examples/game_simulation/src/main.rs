use cli::{Config, GameType};
use rs_poker::holdem::MonteCarloGame;

fn main() {
    let config = Config::new().expect("Failed to get CLI config");
    println!("Starting Hands =\t{:?}", &config.hands);

    let mut g = match config.game_type {
        GameType::MonteCarlo => {
            MonteCarloGame::new(config.hands).expect("Should be able to create a game.")
        }
        GameType::Omaha => unimplemented!(),
    };

    let mut wins: [u64; 2] = [0, 0];

    for _ in 0..config.num_of_games_to_simulate {
        let r = g.simulate();
        g.reset();
        wins[r.0.ones().next().unwrap()] += 1
    }

    let normalized: Vec<f64> = wins
        .iter()
        .map(|cnt| *cnt as f64 / config.num_of_games_to_simulate as f64)
        .collect();

    println!("Wins =\t\t\t{:?}", wins);
    println!("Normalized Wins =\t{:?}", normalized);
}
