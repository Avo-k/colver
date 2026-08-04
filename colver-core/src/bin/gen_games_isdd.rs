//! Génère des **donnes complètes** jouées par IS-DD, au format COLVGM01.
//!
//! Ce que produisent les générateurs existants et ce qui manquait :
//!
//! | binaire | enchère | ce qui est gardé |
//! |---------|---------|------------------|
//! | `gen_pool` | aucune (atout imposé, 4 couleurs) | la valeur DD |
//! | `enrich_pool_isdd` | aucune (`setup_dd`) | les points N-S finaux |
//! | **`gen_games_isdd`** | **réelle, tous les tours** | **toutes les actions** |
//!
//! Les deux premiers jettent les coups : ils produisent une *étiquette* par
//! donne. Pour entraîner un playgen sur du jeu fort il faut la trajectoire —
//! l'enchère telle qu'elle s'est jouée et les 32 cartes dans l'ordre. C'est
//! exactement `COLVGM01`, ce que `train_playgen --games` consomme déjà.
//!
//! ## Concurrence : pourquoi tant de threads pour si peu de cœurs
//!
//! Une décision IS-DD alterne deux régimes qui n'utilisent pas la même
//! ressource : un aller-retour au sidecar (le thread **dort**, le GPU
//! travaille) puis N solves DD (le thread **brûle** un cœur, le GPU dort).
//! Un thread par cœur laisse donc les deux moitiés inoccupées à tour de rôle.
//! Sur-souscrire — `--threads` bien au-dessus de `nproc` — remplit les creux :
//! pendant qu'un thread attend son lot de mondes, un autre résout les siens.
//! C'est aussi ce qui garnit le groupeur du sidecar, dont le coût GPU est
//! quasi constant en nombre de lanes jusqu'au genou de débit.
//!
//! Corollaire : `[play] parallel` doit rester **faux** ici. Il fait résoudre
//! les mondes d'*une* décision sur le pool rayon global, ce qui est le bon
//! choix pour une latence par coup et le mauvais pour un débit — avec T
//! recherches concurrentes le pool est déjà saturé, et le fan-out ne fait
//! qu'ajouter de la contention.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features parallel --bin gen_games_isdd -- \
//!   --bot arena/bots/gen_isdd.toml --deals 10000 --dets 40 --threads 128 \
//!   --out data/training/isdd_games.bin
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::agent::{AgentSpec, MatchContext, Player};
use colver_core::game_loop::MATCH_TARGET;
use colver_core::game_replay::GameReplay;
use colver_core::state::{GameState, Phase};

struct Args {
    bot: String,
    deals: usize,
    dets: Option<u32>,
    /// `"60,60,60,30,30,20,20"` — mondes par décision de 8 cartes restantes à 2.
    dets_schedule: Option<String>,
    threads: usize,
    out: Option<String>,
    url: Option<String>,
    seed: u64,
    /// Enchaîner les donnes en parties de 2000 points au lieu de les tirer
    /// indépendantes. Le score courant part au bidder (v6 est score-aware),
    /// donc le corpus contient alors des enchères de fin de partie — que le
    /// corpus playgen actuel, entièrement à 0-0, ne contient pas du tout.
    match_mode: bool,
    progress_every: usize,
    /// Donnes par éclat intermédiaire. `GameReplay::write_all` n'écrit qu'à la
    /// fin ; une génération de plusieurs heures interrompue à 95 % ne laisserait
    /// rien du tout. Les éclats sont écrits au fil de l'eau, puis fusionnés et
    /// effacés à la fin — ils ne survivent que si le run ne s'est pas terminé.
    shard: usize,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        bot: String::from("arena/bots/gen_isdd.toml"),
        deals: 100,
        dets: None,
        dets_schedule: None,
        threads: 0,
        out: None,
        url: None,
        seed: 42,
        match_mode: false,
        progress_every: 0,
        shard: 5000,
    };
    let mut i = 1;
    while i < argv.len() {
        let next = |i: usize| -> String {
            argv.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("{} attend une valeur", argv[i]);
                std::process::exit(2);
            })
        };
        match argv[i].as_str() {
            "--bot" => { a.bot = next(i); i += 2 }
            "--deals" => { a.deals = next(i).parse().unwrap(); i += 2 }
            "--dets" => { a.dets = Some(next(i).parse().unwrap()); i += 2 }
            "--dets-schedule" => { a.dets_schedule = Some(next(i)); i += 2 }
            "--threads" => { a.threads = next(i).parse().unwrap(); i += 2 }
            "--out" => { a.out = Some(next(i)); i += 2 }
            "--url" => { a.url = Some(next(i)); i += 2 }
            "--seed" => { a.seed = next(i).parse().unwrap(); i += 2 }
            "--match-mode" => { a.match_mode = true; i += 1 }
            "--shard" => { a.shard = next(i).parse().unwrap(); i += 2 }
            // Rassemble les éclats d'un run interrompu en un seul COLVGM01.
            "--merge" => {
                let prefix = next(i);
                let out = argv
                    .iter()
                    .position(|x| x == "--out")
                    .and_then(|p| argv.get(p + 1))
                    .cloned()
                    .unwrap_or_else(|| format!("{prefix}.bin"));
                // Tolérant aux trous, comme le rassemblage interne : un éclat
                // manquant au milieu (écriture ratée) ne doit pas faire
                // silencieusement abandonner tous les suivants. S'arrêter au
                // premier trou et rendre « 0 donne » était le même travail avec
                // la politique inverse.
                let mut all: Vec<GameReplay> = Vec::new();
                let (mut found, mut run, mut last) = (0usize, 0usize, 0usize);
                let mut missing: Vec<usize> = Vec::new();
                for n in 0..10_000 {
                    let p = format!("{prefix}.{n:04}");
                    match GameReplay::load_all(&p) {
                        Ok(mut r) => {
                            found += 1;
                            last = n;
                            run = 0;
                            all.append(&mut r);
                        }
                        Err(_) => {
                            missing.push(n);
                            run += 1;
                            // 64 index absents d'affilée : la série est finie.
                            if run > 64 {
                                break;
                            }
                        }
                    }
                }
                // Seuls les trous *avant* le dernier éclat lu en sont : la
                // queue de la sonde n'est pas un trou, c'est la fin de la série.
                let holes: Vec<usize> = missing.into_iter().filter(|&n| n < last).collect();
                if !holes.is_empty() {
                    eprintln!(
                        "⚠️  {} éclat(s) manquant(s) au milieu de la série : {:?}",
                        holes.len(),
                        &holes[..holes.len().min(10)]
                    );
                }
                println!("{found} éclat(s) lus");
                let bad = verify(&all);
                println!("{} donnes rassemblées, {bad} irrejouables", all.len());
                // Un préfixe qui ne désigne aucun éclat rendait ici un corpus
                // vide, parfaitement « valide » (zéro donne irrejouable), écrit
                // par-dessus `--out`. Une faute de frappe suffisait donc à
                // effacer un corpus existant en sortant 0.
                if all.is_empty() {
                    eprintln!("❌ aucun éclat trouvé en {prefix}.0000 — rien n'est écrit");
                    std::process::exit(1);
                }
                if bad > 0 {
                    std::process::exit(1);
                }
                GameReplay::write_all(&out, &all).expect("écriture");
                println!("→ {out}");
                std::process::exit(0);
            }
            "--progress-every" => { a.progress_every = next(i).parse().unwrap(); i += 2 }
            "--bench" => { a.out = None; i += 1 }
            // Relit un corpus déjà écrit et le rejoue. Ferme l'aller-retour
            // écriture → lecture, que la vérification en mémoire ne couvre pas.
            "--check" => {
                let path = next(i);
                let r = GameReplay::load_all(&path).unwrap_or_else(|e| {
                    eprintln!("{path} : {e}");
                    std::process::exit(1);
                });
                let bad = verify(&r);
                let acts: usize = r.iter().map(|g| g.actions.len()).sum();
                println!(
                    "{path} : {} donnes, {} actions ({:.1}/donne), {bad} irrejouables",
                    r.len(),
                    acts,
                    acts as f64 / r.len().max(1) as f64
                );
                let ok = describe(&r);
                std::process::exit(if bad == 0 && ok { 0 } else { 1 });
            }
            other => { eprintln!("argument inconnu : {other}"); std::process::exit(2) }
        }
    }
    if a.threads == 0 {
        a.threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8) * 4;
    }
    if a.progress_every == 0 {
        a.progress_every = (a.deals / 20).max(1);
    }
    a
}

/// Joue une donne en enregistrant les actions. Copie volontaire de
/// [`colver_core::game_loop::play_deal`] : celui-ci rend le score, pas la
/// trajectoire, et `play_deal_traced` alloue en plus les `Decision` complètes
/// (les `card_scores` de chaque coup) dont on n'a pas l'usage ici.
fn play_and_record(
    state: &mut GameState,
    players: &mut [Box<dyn Player>; 4],
    ctx: &mut MatchContext,
    actions: &mut Vec<u8>,
) -> Result<[i32; 2], colver_core::agent::AgentError> {
    ctx.reset_deal(state.dealer);
    for p in players.iter_mut() {
        p.init_deal(state);
    }
    actions.clear();
    while !state.is_terminal() {
        let seat = state.current_player();
        let before = *state;
        let action = players[seat as usize].action(&before, ctx)?;
        for p in players.iter_mut() {
            p.observe(&before, seat, action);
        }
        ctx.track(&before, action);
        actions.push(action);
        state.step(action);
    }
    let s = state.deal_score();
    Ok([s.scores[0] as i32, s.scores[1] as i32])
}

fn main() {
    let args = parse_args();

    let mut spec = AgentSpec::from_toml_file(&args.bot).unwrap_or_else(|e| {
        eprintln!("spec {} : {e}", args.bot);
        std::process::exit(1);
    });
    if let Some(d) = args.dets {
        spec.play.determinizations = d;
        spec.play.time_ms = 0; // mode compte : D mondes exactement, quel qu'en soit le temps
    }
    if let Some(sched) = &args.dets_schedule {
        spec.play.det_schedule = Some(
            colver_core::agent::spec::parse_det_schedule(sched).unwrap_or_else(|e| {
                eprintln!("--dets-schedule : {e}");
                std::process::exit(2);
            }),
        );
        spec.play.time_ms = 0;
    }
    if let Some(u) = &args.url {
        spec.worlds.url = Some(u.clone());
    }
    // Le fan-out rayon par décision est contre-productif à ce niveau de
    // concurrence (cf. l'en-tête du module) ; on le coupe explicitement plutôt
    // que d'espérer que la spec l'ait fait.
    spec.play.parallel = false;

    eprintln!(
        "gen_games_isdd : bot={} dets={} threads={} donnes={} mode={}",
        spec.label(),
        args.dets_schedule.clone().unwrap_or_else(|| spec.play.determinizations.to_string()),
        args.threads,
        args.deals,
        if args.match_mode { "parties 2000" } else { "donnes indépendantes" },
    );

    // Des éclats d'un run précédent au même chemin seraient écrasés un à un,
    // et le run interrompu qu'ils représentent deviendrait irrécupérable avant
    // même que quiconque sache qu'il existait.
    if let Some(path) = &args.out {
        let leftover: Vec<String> = (0..)
            .map(|n| format!("{path}.{n:04}"))
            .take_while(|p| std::path::Path::new(p).exists())
            .collect();
        if !leftover.is_empty() {
            eprintln!(
                "❌ {} éclat(s) d'un run précédent en {path}.* — les rassembler d'abord :\n\
                 \t--merge {path} --out {path}\n\
                 ou les déplacer. Ils ne seront pas écrasés.",
                leftover.len()
            );
            std::process::exit(1);
        }
    }

    let spec = Arc::new(spec);
    let next_deal = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    // Donnes abandonnées sur erreur de source. Leur jeton est rendu, sinon
    // `--deals N` rendrait N moins les incidents.
    let extra = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));
    // Un hoquet passe, une panne dure. Le budget doit se lire en *secondes de
    // panne*, pas en nombre d'erreurs : sous une coupure totale chaque thread
    // en produit une par expiration de lecture (6 s), donc les erreurs
    // s'accumulent à `threads / 6` par seconde — 43/s à 256 threads. Un budget
    // fixe à 50 abandonnerait après une seconde de coupure. `threads × 4` vaut
    // ~25 s de coupure totale quel que soit le parallélisme, ce qui laisse
    // passer un redémarrage de sidecar sans laisser tourner une nuit contre un
    // GPU mort.
    let error_budget = (args.threads * 4).max(args.deals / 20).max(50);
    let out: Arc<Mutex<Vec<GameReplay>>> = Arc::new(Mutex::new(Vec::with_capacity(args.deals)));
    let start = Instant::now();

    let mut handles = Vec::with_capacity(args.threads);
    for tid in 0..args.threads {
        let spec = spec.clone();
        let next_deal = next_deal.clone();
        let done = done.clone();
        let failed = failed.clone();
        let extra = extra.clone();
        let errors = errors.clone();
        let out = out.clone();
        let total = args.deals;
        let seed = args.seed;
        let match_mode = args.match_mode;
        let progress_every = args.progress_every;
        handles.push(std::thread::spawn(move || {
            // Un jeu de joueurs par thread, construit une fois : les poids sont
            // dans un cache global (`agent::models`), mais l'état par donne et
            // les RNG ne le sont pas.
            let mut players: [Box<dyn Player>; 4] = match (0..4)
                .map(|s| spec.build(s))
                .collect::<Result<Vec<_>, _>>()
            {
                Ok(v) => v.try_into().map_err(|_| ()).expect("4 sièges"),
                Err(e) => {
                    eprintln!("thread {tid} : construction du bot : {e}");
                    failed.store(true, Ordering::Relaxed);
                    return;
                }
            };

            let mut rng = StdRng::seed_from_u64(seed ^ (tid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut ctx = MatchContext::new(0);
            let mut dealer: u8 = rng.gen_range(0..4);
            let mut actions: Vec<u8> = Vec::with_capacity(64);

            loop {
                if failed.load(Ordering::Relaxed) {
                    break;
                }
                let idx = next_deal.fetch_add(1, Ordering::Relaxed);
                if idx >= total + extra.load(Ordering::Relaxed) {
                    break;
                }

                // Une partie terminée remet le compteur à zéro ; hors mode
                // partie, chaque donne est indépendante et le score reste 0-0.
                if !match_mode
                    || ctx.scores[0] >= MATCH_TARGET
                    || ctx.scores[1] >= MATCH_TARGET
                {
                    ctx = MatchContext::new(dealer);
                }

                let mut state = GameState::deal_random(dealer, &mut rng);
                let hands = state.hands;
                match play_and_record(&mut state, &mut players, &mut ctx, &mut actions) {
                    Ok(score) => {
                        if match_mode {
                            ctx.scores[0] += score[0];
                            ctx.scores[1] += score[1];
                        }
                        out.lock().unwrap().push(GameReplay {
                            dealer,
                            hands,
                            actions: actions.clone(),
                        });
                    }
                    Err(e) => {
                        // Une erreur de source **ne tue pas le run**. Sous
                        // `fallback = "strict"` — le seul réglage honnête pour
                        // un corpus — un aller-retour qui expire rend une
                        // erreur, et un GPU momentanément saturé en produit
                        // plusieurs d'un coup. Faire tomber la génération
                        // entière là-dessus, c'est perdre des heures de travail
                        // sur un hoquet de quelques secondes : mesuré, un run
                        // de 28 000 donnes s'est arrêté à 5 076 parce qu'un
                        // second processus s'est mis à partager le GPU.
                        //
                        // La donne en cours est **jetée** — jamais enregistrée
                        // à moitié — et son jeton rendu, pour que `--deals N`
                        // reste un compte de donnes complètes. Le run ne
                        // s'arrête que si le budget d'erreurs saute, c'est-à-dire
                        // si la panne dure au lieu de passer.
                        let n = errors.fetch_add(1, Ordering::Relaxed) + 1;
                        extra.fetch_add(1, Ordering::Relaxed);
                        if n <= 5 || n % 100 == 0 {
                            eprintln!("thread {tid} donne {idx} abandonnée ({n}e erreur) : {e}");
                        }
                        if n > error_budget {
                            eprintln!(
                                "❌ {n} erreurs de source dépassent le budget ({error_budget}) —                                  la panne dure, arrêt"
                            );
                            failed.store(true, Ordering::Relaxed);
                            break;
                        }
                        continue;
                    }
                }
                dealer = (dealer + 1) % 4;

                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n % progress_every == 0 || n == total {
                    let el = start.elapsed().as_secs_f64();
                    let rate = n as f64 / el;
                    eprintln!(
                        "  {n}/{total} donnes  {rate:.2}/s  {el:.0}s écoulées  ETA {:.0}s",
                        (total - n) as f64 / rate.max(1e-9)
                    );
                }
            }
        }));
    }

    // Videur d'éclats : draine le tampon partagé au fil de l'eau pour qu'une
    // interruption ne coûte que le dernier éclat. N'écrit rien sans `--out`.
    let shard_prefix = args.out.clone();
    let stop = Arc::new(AtomicBool::new(false));
    let shard_n = Arc::new(AtomicUsize::new(0));
    let lost = Arc::new(AtomicBool::new(false));
    let flusher = {
        let out = out.clone();
        let stop = stop.clone();
        let shard_n = shard_n.clone();
        let lost = lost.clone();
        let want = args.shard;
        std::thread::spawn(move || {
            let Some(prefix) = shard_prefix else { return };
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3));
                let last = stop.load(Ordering::Relaxed);
                let batch: Vec<GameReplay> = {
                    let mut g = out.lock().unwrap();
                    if g.len() < want && !last {
                        continue;
                    }
                    std::mem::take(&mut *g)
                };
                if !batch.is_empty() {
                    let n = shard_n.fetch_add(1, Ordering::Relaxed);
                    let path = format!("{prefix}.{n:04}");
                    if let Err(e) = GameReplay::write_all(&path, &batch) {
                        // Les donnes de cet éclat n'existent plus nulle part :
                        // le tampon a été vidé pour les écrire. On ne peut pas
                        // les récupérer — on peut seulement refuser de faire
                        // passer le corpus tronqué pour complet.
                        eprintln!("éclat {path} : {e} — {} donnes perdues", batch.len());
                        lost.store(true, Ordering::Relaxed);
                    }
                }
                if last {
                    return;
                }
            }
        })
    };

    for h in handles {
        // Un thread qui panique rend `Err` ici. Sans ce test le run écrivait un
        // corpus court au chemin de sortie normal et sortait 0 — la panique
        // n'apparaissant que dans le journal, que personne ne relit quand tout
        // « a marché ».
        if h.join().is_err() {
            eprintln!("un thread a paniqué");
            failed.store(true, Ordering::Relaxed);
        }
    }
    stop.store(true, Ordering::Relaxed);
    let _ = flusher.join();

    let elapsed = start.elapsed().as_secs_f64();
    // Les éclats sont la source de vérité : le tampon a été vidé dedans.
    let replays: Vec<GameReplay> = match &args.out {
        Some(prefix) => {
            let mut all = Vec::new();
            for n in 0..shard_n.load(Ordering::Relaxed) {
                let p = format!("{prefix}.{n:04}");
                match GameReplay::load_all(&p) {
                    Ok(mut r) => all.append(&mut r),
                    Err(e) => {
                        eprintln!("relecture {p} : {e}");
                        lost.store(true, Ordering::Relaxed);
                    }
                }
            }
            all
        }
        None => Arc::try_unwrap(out).ok().unwrap().into_inner().unwrap(),
    };

    let n_err = errors.load(Ordering::Relaxed);
    if n_err > 0 {
        eprintln!(
            "\n⚠️  {n_err} donne(s) abandonnée(s) sur erreur de source (budget {error_budget})"
        );
    }
    if failed.load(Ordering::Relaxed) {
        eprintln!("⚠️  arrêt sur erreur — {} donnes complètes malgré tout", replays.len());
    }

    // ── Profil ────────────────────────────────────────────────────────────
    let decisions: usize = replays.iter().map(|r| r.actions.len()).sum();
    eprintln!(
        "\n{} donnes en {:.1}s — {:.2} donnes/s, {:.0} actions/s",
        replays.len(),
        elapsed,
        replays.len() as f64 / elapsed,
        decisions as f64 / elapsed,
    );
    print_profile(elapsed, args.threads);

    if let Some(path) = &args.out {
        let bad = verify(&replays);
        if bad > 0 {
            eprintln!(
                "\n❌ {bad} donnes sur {} ne se rejouent pas — rien n'est écrit",
                replays.len()
            );
            std::process::exit(1);
        }
        GameReplay::write_all(path, &replays).expect("écriture COLVGM01");
        let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        // Le fichier fusionné existe et se rejoue : les éclats n'ont plus de
        // raison d'être. On ne les efface qu'ici, jamais avant — et jamais du
        // tout si quelque chose s'est perdu en route, parce qu'ils sont alors
        // la seule copie de ce qui reste.
        let broken = lost.load(Ordering::Relaxed) || failed.load(Ordering::Relaxed);
        if !broken {
            for n in 0..shard_n.load(Ordering::Relaxed) {
                let _ = std::fs::remove_file(format!("{path}.{n:04}"));
            }
        }
        eprintln!(
            "→ {path} : {} donnes, {:.1} Mo ({:.0} o/donne)",
            replays.len(),
            bytes as f64 / 1e6,
            bytes as f64 / replays.len().max(1) as f64,
        );
        // Un corpus court doit se voir au code de retour, pas seulement dans un
        // journal. `--deals N` est une commande, pas un souhait : en rendre
        // moins sans le dire, c'est exactement le mode de défaillance que la
        // vérification de légalité est censée fermer, transposé au compte.
        if broken || replays.len() != args.deals {
            // Distinguer les deux causes : un run qui s'est ARRÊTÉ n'a rien
            // perdu, il n'a simplement pas tout produit — et le message le
            // disait quand même, ce qui envoyait chercher des donnes disparues
            // là où il n'y avait qu'une URL morte.
            let cause = if lost.load(Ordering::Relaxed) {
                " — des donnes écrites ont été perdues"
            } else if failed.load(Ordering::Relaxed) {
                " — le run s'est arrêté sur une erreur (voir plus haut)"
            } else {
                ""
            };
            eprintln!(
                "❌ corpus INCOMPLET : {} donnes sur {} demandées{cause}. Les éclats sont conservés.",
                replays.len(),
                args.deals,
            );
            std::process::exit(1);
        }
    }
}

/// Rejoue chaque donne enregistrée et compte celles qui ne tiennent pas.
///
/// `GameState::step` **ne valide pas la légalité** — c'est le contrat d'un
/// moteur RL, et c'est ce qui a laissé six donnes fausses entrer en base côté
/// web (`integrity.py`). Ici l'enjeu est plus grand : une donne incohérente
/// dans un corpus d'entraînement n'est pas un incident visible, c'est du
/// gradient sur une position qui n'existe pas. On vérifie donc à l'écriture,
/// exactement comme le web vérifie à l'enregistrement : chaque action légale à
/// son tour, et la dernière rend la donne terminale.
fn verify(replays: &[GameReplay]) -> usize {
    let mut bad = 0;
    for r in replays {
        let mut s = GameState::new(r.dealer, r.hands);
        let mut ok = true;
        for &a in &r.actions {
            if s.is_terminal() || s.legal_actions() & (1u64 << a) == 0 {
                ok = false;
                break;
            }
            s.step(a);
        }
        if !ok || !s.is_terminal() {
            bad += 1;
        }
    }
    bad
}

/// Ce qu'il y a *dans* le corpus, pas seulement qu'il se relit.
///
/// Rend `false` sur un corpus vide : c'est le cas le plus dégénéré possible, et
/// c'était justement celui sur lequel cette fonction restait muette.
///
/// Un corpus peut être parfaitement rejouable et pourtant inutilisable : rien
/// n'interdit à un joueur mal configuré de passer neuf donnes sur dix, et une
/// donne passée ne contient aucune carte. La vérification de format ne le
/// verrait pas — d'où ce résumé, qui décrit la *distribution* et pas la
/// structure.
fn describe(r: &[GameReplay]) -> bool {
    if r.is_empty() {
        eprintln!("❌ corpus VIDE — 0 donne");
        return false;
    }
    let mut passed = 0usize;
    let mut by_value: std::collections::BTreeMap<u8, usize> = Default::default();
    let mut by_suit = [0usize; 4];
    let mut coinched = 0usize;
    let mut belote = 0usize;
    let mut taker_ns = 0usize;
    let mut made = 0usize;
    let mut bid_actions = 0usize;

    for g in r {
        let mut s = GameState::new(g.dealer, g.hands);
        for &a in &g.actions {
            if s.phase == Phase::Bidding {
                bid_actions += 1;
            }
            s.step(a);
        }
        if s.contract.value == 0 {
            passed += 1;
            continue;
        }
        *by_value.entry(s.contract.value).or_default() += 1;
        by_suit[s.contract.trump as usize] += 1;
        if s.contract.coinche > 0 {
            coinched += 1;
        }
        if s.belote != [0, 0] {
            belote += 1;
        }
        let taker = s.contract.team;
        if taker == 0 {
            taker_ns += 1;
        }
        let sc = s.deal_score();
        if sc.scores[taker as usize] > 0 {
            made += 1;
        }
    }
    let played = r.len() - passed;
    println!(
        "  {} donnes jouées, {passed} passées ({:.1} %) — {:.1} annonces par donne",
        played,
        100.0 * passed as f64 / r.len() as f64,
        bid_actions as f64 / r.len() as f64,
    );
    if played == 0 {
        eprintln!("❌ aucune donne jouée — toutes passées");
        return false;
    }
    let pct = |n: usize| 100.0 * n as f64 / played as f64;
    println!(
        "  preneur N-S {:.1} %  ·  contrat réussi {:.1} %  ·  contré {:.1} %  ·  belote {:.1} %",
        pct(taker_ns), pct(made), pct(coinched), pct(belote),
    );
    let suits = ["♠", "♥", "♦", "♣"];
    let s: Vec<String> = (0..4).map(|i| format!("{} {:.0} %", suits[i], pct(by_suit[i]))).collect();
    println!("  couleurs : {}", s.join("  "));
    let v: Vec<String> = by_value
        .iter()
        .map(|(val, n)| format!("{val} : {:.0} %", pct(*n)))
        .collect();
    println!("  contrats : {}", v.join("  "));
    true
}

/// Où passe le temps, par nombre de cartes restantes.
///
/// L'axe « cartes restantes » n'est pas décoratif : entre l'entame et la
/// finale le coût d'un solve varie de quatre ordres de grandeur et le coût
/// d'un monde playgen d'un facteur ~8 (autant de pas de décodage que de
/// cartes cachées). Un agrégat sur la donne mélangerait les deux régimes.
fn print_profile(wall: f64, threads: usize) {
    use colver_core::agent::isdd::telemetry;
    let lanes = telemetry::lanes();
    if lanes.is_empty() {
        return;
    }
    let s = telemetry::snapshot();
    eprintln!(
        "\ndécisions IS-DD : {} — {} sans échantillonnage, {} 100 % playgen, {} partielles, {} sans playgen",
        s.decisions, s.no_sampling, s.all_playgen, s.partial, s.no_playgen
    );
    eprintln!(
        "mondes : {} injectés, {} belief, {} uniformes ({:.1} % de repli)",
        s.worlds_injected, s.worlds_belief, s.worlds_uniform, s.fallback_world_pct()
    );

    eprintln!("\n cartes  décisions   mondes   tours   source ms   solve ms   attente%   ms/tour  remplissage");
    let (mut tot_src, mut tot_all) = (0u64, 0u64);
    for l in &lanes {
        let solve_us = l.total_us.saturating_sub(l.source_us);
        tot_src += l.source_us;
        tot_all += l.total_us;
        eprintln!(
            "   {:>2}    {:>8}  {:>7}  {:>6}  {:>9.0}  {:>9.0}   {:>7.1}  {:>8.1}   {:>6.1}%",
            l.cards_left,
            l.decisions,
            l.solved,
            l.rounds,
            l.source_us as f64 / 1e3,
            solve_us as f64 / 1e3,
            l.wait_pct(),
            l.ms_per_round(),
            l.fill_pct(),
        );
    }
    let tot_solve = tot_all.saturating_sub(tot_src);
    // `tot_all` est du temps *cumulé sur les threads*. Rapporté au mur × threads
    // il dit quelle part de la capacité offerte a réellement servi : le reste
    // est du temps de thread perdu ailleurs (ordonnancement, allocation, GC de
    // l'OS), et un écart important signale que la sur-souscription ne paie plus.
    let capacity = wall * threads as f64 * 1e6;
    eprintln!(
        "\ntotal thread : {:.0} s attente sidecar + {:.0} s solve DD = {:.0} s sur {:.0} s de capacité ({} threads × {:.0} s) — {:.0} % utilisés",
        tot_src as f64 / 1e6,
        tot_solve as f64 / 1e6,
        tot_all as f64 / 1e6,
        capacity / 1e6,
        threads,
        wall,
        100.0 * tot_all as f64 / capacity,
    );
    eprintln!(
        "part attente sidecar : {:.1} %   part solve DD : {:.1} %",
        100.0 * tot_src as f64 / tot_all.max(1) as f64,
        100.0 * tot_solve as f64 / tot_all.max(1) as f64,
    );
    let _ = Phase::Done;
}
