//! Combien de **classes de mains** un pool de donnes couvre-t-il ?
//!
//! [docs/data_gen/isdd_score_layer_v2.md](../../../docs/data_gen/isdd_score_layer_v2.md) §11
//! nomme le risque sans le chiffrer : échanger 5 M d'étiquettes faibles contre 500 k
//! fortes divise par dix le nombre de mains vues, et *« la couverture de classes n'est
//! plus acquise »*. Ce binaire la mesure.
//!
//! Une main de 8 cartes appartient à l'une des **472 579** classes d'équivalence par
//! permutation de couleurs (`hand_class_id`, bijection testée). Avant qu'un atout soit
//! nommé, deux mains de la même classe **sont** la même main : c'est l'espace dans lequel
//! une politique d'ouverture est une table.
//!
//! Pourquoi le coupon collector induit en erreur ici, et dans quel sens. Il demande
//! ~6,2 M de tirages pour voir **toutes** les classes, ce qui a fait craindre que 2 M de
//! mains n'en couvrent qu'une fraction. Mais son estimation porte sur le **dernier**
//! coupon, pas sur les 97 premiers pourcents — et mesuré, la distribution est presque
//! **uniforme** (le centile le plus fréquent porte 2,5 % des mains contre 1,0 % à
//! l'uniforme exacte), donc la couverture monte vite et ne traîne qu'à la toute fin.
//! Le contre-pied de l'intuition est dans ce sens-là, pas dans l'autre.
//!
//! ```bash
//! ./target/release/bench_class_coverage --pool data/deals/base_5M.bin --steps 100000,500000,5000000
//! ```

use std::collections::HashMap;

use colver_core::hand_class::{hand_class_id, NUM_HAND_CLASSES};

fn main() {
    let mut pool = String::from("data/deals/base_5M.bin");
    let mut steps: Vec<usize> = vec![100_000, 500_000, 1_000_000, 5_000_000];
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" => { i += 1; pool = args[i].clone() }
            "--steps" => {
                i += 1;
                steps = args[i].split(',').map(|s| s.parse().unwrap()).collect();
            }
            "--help" | "-h" => {
                eprintln!("bench_class_coverage : classes de mains couvertes par un pool");
                eprintln!("  --pool <path>    COLVDD01");
                eprintln!("  --steps a,b,c    paliers de donnes où relever la couverture");
                std::process::exit(0)
            }
            o => { eprintln!("argument inconnu : {o}"); std::process::exit(1) }
        }
        i += 1;
    }

    // Lecture directe du format : charger un `DealPool` de 5 M alloue 105 Mo de
    // structures dont on n'utilise que les mains.
    let data = std::fs::read(&pool).expect("lecture du pool");
    if &data[..8] != b"COLVDD01" {
        eprintln!("{pool} : magic inattendu");
        std::process::exit(1);
    }
    let total = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    eprintln!("{pool} : {total} donnes, {NUM_HAND_CLASSES} classes possibles");

    let mut seen: HashMap<u32, u32> = HashMap::with_capacity(1 << 20);
    let mut next_step = 0usize;
    steps.sort_unstable();
    println!("\n {:>10} {:>12} {:>9} {:>12} {:>10}",
             "donnes", "mains", "classes", "couverture", "vues 1 fois");
    for k in 0..total {
        let off = 16 + 21 * k;
        for h in 0..4 {
            let hand = u32::from_le_bytes(
                data[off + 1 + 4 * h..off + 5 + 4 * h].try_into().unwrap());
            *seen.entry(hand_class_id(hand)).or_insert(0) += 1;
        }
        if next_step < steps.len() && k + 1 == steps[next_step] {
            let singles = seen.values().filter(|&&c| c == 1).count();
            println!(
                " {:>10} {:>12} {:>9} {:>11.2}% {:>9.1}%",
                k + 1, 4 * (k + 1), seen.len(),
                100.0 * seen.len() as f64 / NUM_HAND_CLASSES as f64,
                100.0 * singles as f64 / seen.len() as f64,
            );
            next_step += 1;
        }
        if next_step >= steps.len() {
            break;
        }
    }

    // La couverture brute ne dit pas si le modèle a de quoi apprendre : une classe vue
    // une seule fois donne un gradient, pas une statistique. On regarde donc combien de
    // classes passent des seuils utilisables.
    let mut counts: Vec<u32> = seen.values().copied().collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));
    // ⚠️ Ce tableau porte sur le DERNIER palier parcouru, pas sur tous. L'omettre ferait
    // lire des occurrences de 5 M de donnes comme si elles valaient pour 500 k.
    let at = steps.last().copied().unwrap_or(total).min(total);
    println!("\n classes atteignant un nombre d'occurrences — À {at} DONNES :");
    for t in [1u32, 2, 5, 10, 50, 100] {
        let c = counts.iter().filter(|&&x| x >= t).count();
        println!("   ≥ {t:>3} fois : {c:>8}  ({:.2} % des classes possibles)",
                 100.0 * c as f64 / NUM_HAND_CLASSES as f64);
    }
    let tot: u64 = counts.iter().map(|&x| x as u64).sum();
    let top1 = counts.len() / 100;
    let mass: u64 = counts[..top1.max(1)].iter().map(|&x| x as u64).sum();
    println!("\n le centile le plus fréquent des classes porte {:.1} % des mains",
             100.0 * mass as f64 / tot as f64);
    println!(" (si ce chiffre est grand, la distribution est très inégale et la");
    println!("  couverture brute surestime ce que le modèle voit vraiment)");
}
