//! Empreinte des sources qui décident de ce que le sidecar playgen produit.
//!
//! **Le problème que ça résout.** Le sidecar est un déploiement *manuel*,
//! séparé du webhook (cf. `docs/belief/playgen.md`). Décider s'il est à jour se
//! faisait donc à la lecture des titres de commits — et le 2026-08-03 un commit
//! titré « feat(elo) » a livré la contrainte belote **dans le sampler playgen**
//! sans que rien dans son message ne le dise. Résultat : 21 h de sidecar périmé
//! fabriquant des mondes que `worlds::retain_valid` rejetait ensuite (~15,4 %
//! aux positions à belote). Correct, mais Dédé cherchait sur moins de mondes
//! qu'il n'en demandait, en silence.
//!
//! **Pourquoi une empreinte de sources et pas le SHA git.** Le SHA du dépôt
//! change à *chaque* commit, y compris ceux qui ne touchent que le web. Une
//! alerte qui s'allume à chaque déploiement est du bruit, et une alerte qui est
//! du bruit ne se lit plus — elle serait pire que rien. L'empreinte ci-dessous
//! ne bouge que quand le comportement du sidecar peut bouger.
//!
//! **La surface, et pourquoi ces deux répertoires.** `playgen/` (tokenizer,
//! modèle, génération CPU et GPU) et `engine/` (les règles que les jetons
//! encodent — dont `play.rs::belote_facts`, précisément le fichier manqué).
//! Volontairement **exclus** : `worlds.rs`, qui tourne côté conteneur web et
//! non dans le sidecar — l'y mettre créerait une fausse alerte à chaque fois
//! que le filtrage client bouge ; et le reste de `colver-core`, que le sampler
//! n'atteint pas. La règle est « ces deux répertoires en entier » plutôt qu'une
//! liste de fichiers : une liste dérive, un répertoire s'audite d'un coup d'œil.
//!
//! **Ce que ça ne couvre pas**, et qu'il ne faut pas lui faire dire : les poids
//! du checkpoint (vérifiés à part, par sha256 — cf. la même doc), les drapeaux
//! de compilation, la version de CUDA. La question à laquelle cette empreinte
//! répond est exactement « le sidecar a-t-il été construit sur les mêmes
//! sources playgen/engine que le conteneur web ? », rien de plus.
//!
//! FNV-1a 64 bits : on détecte une dérive accidentelle entre deux machines de
//! confiance, on ne se défend pas contre un adversaire. Pas de dépendance, ce
//! qui préserve le « zéro dépendance par défaut » de `colver-core`.

use std::fs;

/// Les répertoires dont le contenu décide du comportement du sidecar.
const SURFACE_DIRS: [&str; 2] = ["src/playgen", "src/engine"];

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_1a(state: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *state ^= u64::from(b);
        *state = state.wrapping_mul(FNV_PRIME);
    }
}

fn main() {
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();

    for dir in SURFACE_DIRS {
        // Sur le répertoire *et* sur chaque fichier : sans la ligne sur le
        // répertoire, l'ajout ou la suppression d'un fichier ne relancerait
        // pas ce script et l'empreinte mentirait.
        println!("cargo:rerun-if-changed={dir}");

        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => panic!("build.rs : impossible de lire {dir} : {e}"),
        };

        for entry in entries {
            let path = entry.expect("entrée de répertoire").path();
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            println!("cargo:rerun-if-changed={}", path.display());
            let bytes = fs::read(&path)
                .unwrap_or_else(|e| panic!("build.rs : lecture de {} : {e}", path.display()));
            files.push((path.to_string_lossy().replace('\\', "/"), bytes));
        }
    }

    // L'ordre de `read_dir` dépend du système de fichiers : sans ce tri, deux
    // machines aux sources identiques produiraient des empreintes différentes,
    // ce qui est le pire des résultats — une fausse alerte permanente.
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut state = FNV_OFFSET;
    for (name, bytes) in &files {
        // Le nom entre dans l'empreinte : renommer ou supprimer un fichier doit
        // se voir autant qu'en modifier un.
        fnv_1a(&mut state, name.as_bytes());
        fnv_1a(&mut state, &[0xff]);

        // `\r` retiré : un checkout Windows ne doit pas se lire comme une
        // dérive de code. Aucune source Rust d'ici n'en contient légitimement.
        let normalized: Vec<u8> = bytes.iter().copied().filter(|&b| b != b'\r').collect();

        // La longueur entre aussi : sans elle, déplacer du texte d'un fichier
        // vers le suivant laisserait la concaténation inchangée.
        fnv_1a(&mut state, &(normalized.len() as u64).to_le_bytes());
        fnv_1a(&mut state, &normalized);
        fnv_1a(&mut state, &[0xfe]);
    }

    println!("cargo:rustc-env=COLVER_PLAYGEN_SURFACE={state:016x}");
}
