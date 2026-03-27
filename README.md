# rs-poker

[![Crates.io](https://img.shields.io/crates/v/rs-poker.svg)](https://crates.io/crates/rs-poker)
[![Docs.rs](https://docs.rs/rs_poker/badge.svg)](https://docs.rs/rs_poker)
[![License](https://img.shields.io/crates/l/rs-poker)](https://github.com/elliottneilclark/rs-poker/blob/master/LICENSE)
[![CI](https://github.com/elliottneilclark/rs-poker/actions/workflows/main.yml/badge.svg)](https://github.com/elliottneilclark/rs-poker/actions/workflows/main.yml)

The most advanced open-source poker library available. Hand evaluation at 50M+ hands/sec per core, a CFR solver for game-theory optimal strategy approximation, Omaha support (PLO4-PLO7), multi-agent arena simulation, Monte Carlo equity calculation, and ICM tournament simulation — all in Rust.

Available as both a library (`rs_poker`) and a CLI binary (`rsp`).

## Quick Start — `rsp` CLI

Install the binary:

```bash
cargo install rs_poker --features rsp
```

Rank a hand:

```bash
rsp holdem rank AcKdQhJsTs
```

Monte Carlo equity simulation (3M hands by default):

```bash
rsp holdem simulate AdKh 8c8s
```

Calculate outs with board cards:

```bash
rsp holdem outs AcAd KsKh --board QhJhTh --detailed
```

Rank an Omaha hand:

```bash
rsp omaha rank AhAsKhKs QhJhTh
```

Pit agents against each other:

```bash
rsp arena compare ./examples/configs -n 5000 -p 3 --parallel 8
```

Generate hand histories in Open Hand History format:

```bash
rsp arena generate ./examples/configs -o hands.ohh -n 1000
```

Convert chipEV to $EV in tournaments:

```bash
rsp icm simulate --chip-stacks 1500 1250 600 500 850 800 --payments 50 30 20 --iterations 50000
```

## Library Usage

Add to your project:

```bash
cargo add rs_poker
```

### Parse and rank a hand

```rust
use rs_poker::core::{FlatHand, Rankable};

let hand = FlatHand::new_from_str("AcKdQhJsTs").unwrap();
let rank = hand.rank();
println!("{}", rank); // StraightFlush
```

### Monte Carlo equity simulation

```rust
use rs_poker::core::{Card, Hand, Suit, Value};
use rs_poker::holdem::MonteCarloGame;

let hero = Hand::new_with_cards(vec![
    Card::new(Value::Ace, Suit::Spade),
    Card::new(Value::Ace, Suit::Heart),
]);
let villain = Hand::new_with_cards(vec![
    Card::new(Value::King, Suit::Spade),
    Card::new(Value::King, Suit::Heart),
]);

let mut sim = MonteCarloGame::new(vec![hero, villain]).unwrap();
let equity = sim.estimate_equity(100_000);
// equity[0] ≈ 0.82 (AA vs KK)
```

### Omaha hand evaluation

```rust
use rs_poker::core::Rankable;
use rs_poker::omaha::OmahaHand;

let hand = OmahaHand::new_from_str("AhAsKhKs", "QhJhTh").unwrap();
let rank = hand.rank();
println!("{}", rank); // StraightFlush — uses exactly 2 hole + 3 board
```

## Core

Foundational types for all poker variants:

- **Card**, **Suit**, **Value** — standard 52-card representations
- **Hand** — bitset-backed hand with O(1) card operations via `CardBitSet` (u64)
- **FlatHand** — vector-backed hand optimized for ranking
- **Deck** — shuffleable deck with dead card support
- **Rank** — full hand ranking: HighCard through StraightFlush with accurate kicker comparison
- **Rankable** trait — unified ranking API implemented by all hand types
- **PlayerBitSet** (u16) — O(1) player tracking for up to 16 players

### Performance

Hand evaluation runs at **50M+ hands/sec per core** (~20ns per 5-card hand, <25ns per 7-card hand). This is achieved through:

- **CardBitSet** (u64) for O(1) card membership, intersection, and union
- **Gosper's hack** for zero-allocation combinatorial iteration
- **PDEP** hardware acceleration on x86_64 for bit manipulation
- Accurate kicker comparison — no single-kicker shortcuts that break tie resolution

## Omaha

Full support for Omaha variants: **PLO4, PLO5, PLO6, and PLO7**.

- Enforces the "exactly 2 hole cards + exactly 3 board cards" rule
- Uses the same `Rankable` trait as Hold'em — one API for all variants
- Evaluates all valid hole+board combinations to find the best hand

```rust
use rs_poker::core::Rankable;
use rs_poker::omaha::OmahaHand;

// PLO5 with 5 hole cards
let hand = OmahaHand::new_from_str("AhAsKhKs9d", "QhJhTh").unwrap();
let rank = hand.rank();
```

## Hold'em

Texas Hold'em-specific tools:

- **Starting hands** — all 169 unique hands (1,326 total combos)
- **Range parsing** — `"AKs"`, `"TT+"`, `"JTs-67s"`, `"KQo+"`
- **Monte Carlo simulation** — multi-player equity estimation via `MonteCarloGame`
- **Outs calculation** — exhaustive enumeration of remaining cards

```rust
use rs_poker::holdem::StartingHand;

// Parse a range of hands
let hands = StartingHand::new_from_range("TT+").unwrap();
// Returns: TT, JJ, QQ, KK, AA
```

## Arena

Multi-agent simulation framework for autonomous poker play. Supports 2-16 players with configurable blinds, antes, and stack sizes.

### Agent Trait

Implement `act()` to create a custom strategy:

```rust
use rs_poker::arena::{Agent, AgentAction, GameState};

struct MyAgent;

impl Agent for MyAgent {
    fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
        AgentAction::Call(game_state.current_bet())
    }
    fn name(&self) -> &str { "my-agent" }
}
```

### Built-in Agents

| Agent | Strategy |
|-------|----------|
| `CallingAgent` | Always calls |
| `FoldingAgent` | Always folds (checks when possible) |
| `AllInAgent` | Always goes all-in |
| `RandomAgent` | Configurable fold/call probabilities per street |
| `RandomPotControlAgent` | Random with pot-size-aware bet limiting |
| `ConfigAgentBuilder` | JSON-configurable preflop ranges + postflop strategy |
| `VecReplayAgent` | Replays a recorded action sequence |
| `CFRAgent` | Counterfactual Regret Minimization solver |

### Historians

Event recorders that observe every action during simulation:

| Historian | Purpose |
|-----------|---------|
| `VecHistorian` | Collects all actions in memory |
| `StatsTrackingHistorian` | Aggregates VPIP, PFR, 3-bet, win rate, positional profit |
| `DirectoryHistorian` | Writes hand histories to disk |
| `OpenHandHistoryHistorian` | Exports in OHH format |
| `FnHistorian` | Closure-based custom recording |
| `NullHistorian` | No-op (discards all records) |

### Competitions

- **`HoldemCompetition`** — run multiple cash game hands and track per-player P&L, win/loss counts, and per-street statistics
- **`SingleTableTournament`** — elimination-style tournament with finishing positions and stack tracking

## CFR Solver

The CFR (Counterfactual Regret Minimization) agent learns game-theory optimal strategies by exploring the game tree and minimizing regret over iterations.

- **PCFR+** regret matching with quadratic weighting
- **Lock-free concurrent tree** — shared across agents via `Arc`
- **Arena allocator** — nodes stored in `Vec` by index for cache-friendly traversal
- **Parallel exploration** via rayon at configurable depth
- **Depth-based iteration scheduling** — more iterations at deeper nodes where the tree is smaller
- **Configurable action generators** — per-street bet sizing (raise multipliers, pot fractions, all-in)
- **Graphviz export** for decision tree visualization

Run CFR agents via the CLI:

```bash
rsp arena compare ./examples/configs -n 5000 -p 3 --parallel 8
```

Or configure them via JSON:

```json
{
  "type": "cfr_configurable",
  "name": "CFR-Agent",
  "depth_hands": [24, 3, 1],
  "action_config": {
    "preflop": {
      "call_enabled": true,
      "raise_mult": [4.0],
      "pot_mult": [],
      "setup_shove": false,
      "all_in": true
    },
    "flop": {
      "call_enabled": true,
      "raise_mult": [],
      "pot_mult": [0.5, 1.0],
      "setup_shove": false,
      "all_in": true
    }
  }
}
```

## Simulated ICM

Monte Carlo-based ICM (Independent Chip Model) tournament equity calculation.

```rust
use rs_poker::simulated_icm::simulate_icm_tournament;

let chip_stacks = vec![10000, 5000, 3000, 2000];
let payments = vec![100, 50, 25, 10];
let winnings = simulate_icm_tournament(&chip_stacks, &payments);
// winnings[i] = expected payout for player i based on chip stack
```

Avoids the O(N!) complexity of traditional ICM by simulating all-in confrontations. Parallelizable across independent tournaments.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `arena` | Yes | Multi-agent simulation framework, CFR solver |
| `serde` | Yes | JSON serialization for hands, game states, configs |
| `omaha` | Yes | Omaha (PLO4/PLO5/PLO6/PLO7) hand evaluation |
| `rsp` | No | CLI binary with all features enabled |
| `open-hand-history` | No | Open Hand History format import/export |
| `arena-test-util` | No | Approximate comparison helpers for tests |
| `open-hand-history-test-util` | No | OHH format testing helpers |

## Testing and Correctness

- **Fuzzing** — hand ranking, game simulation replay, CFR agent, and Omaha evaluation all have fuzz targets
- **Mutation testing** — `cargo-mutants` validates test coverage catches real bugs
- **Benchmarks** — comprehensive suite covering hand ranking, Monte Carlo simulation, arena games, CFR, ICM, and Omaha

## License

Apache-2.0
