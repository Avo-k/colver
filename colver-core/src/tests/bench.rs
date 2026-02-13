use colver_core::rollout;
use colver_core::state::GameState;
use rand::Rng;
use std::time::Instant;

fn main() {
    let mut rng = rand::thread_rng();

    // Benchmark: random rollouts from a mid-game state
    let n = 1_000_000;

    // First, create a state partway through a game (after bidding, a few tricks played)
    let mut state = GameState::deal_random(0, &mut rng);
    // Play through bidding with random moves
    while !state.is_terminal() && state.phase == colver_core::state::Phase::Bidding {
        let legal = state.legal_actions();
        let count = legal.count_ones();
        let idx = rng.gen_range(0..count);
        let action = select_nth_bit(legal, idx);
        state.step(action);
    }

    if state.is_terminal() {
        // 4 passes - try again
        state = GameState::deal_random(0, &mut rng);
        // Force a bid
        state.step(1); // 80 Spades
        state.step(0); // pass
        state.step(0); // pass
        state.step(0); // pass
    }

    println!("State: {:?}", state);
    println!("Phase: {:?}", state.phase);
    println!();

    // Benchmark rollouts
    let start = Instant::now();
    let avg = rollout::rollout_batch(&state, n, &mut rng);
    let elapsed = start.elapsed();

    println!("Ran {} rollouts in {:?}", n, elapsed);
    println!(
        "Rate: {:.2} rollouts/sec",
        n as f64 / elapsed.as_secs_f64()
    );
    println!("Average rewards: NS={:.1}, EW={:.1}", avg[0], avg[1]);
    println!(
        "Per rollout: {:.0} ns",
        elapsed.as_nanos() as f64 / n as f64
    );

    // Also benchmark from fresh deal (includes bidding)
    let start = Instant::now();
    let mut total = [0.0f32; 2];
    for _ in 0..n {
        let mut s = GameState::deal_random(0, &mut rng);
        let r = rollout::rollout_random(&mut s, &mut rng);
        total[0] += r[0];
        total[1] += r[1];
    }
    let elapsed = start.elapsed();
    println!();
    println!("Full deal (including bidding + deal) rollouts:");
    println!("Ran {} rollouts in {:?}", n, elapsed);
    println!(
        "Rate: {:.2} rollouts/sec",
        n as f64 / elapsed.as_secs_f64()
    );
    println!(
        "Per rollout: {:.0} ns",
        elapsed.as_nanos() as f64 / n as f64
    );
}

fn select_nth_bit(mask: u64, mut n: u32) -> u8 {
    let mut remaining = mask;
    loop {
        let bit = remaining.trailing_zeros() as u8;
        if n == 0 {
            return bit;
        }
        n -= 1;
        remaining &= remaining - 1;
    }
}
