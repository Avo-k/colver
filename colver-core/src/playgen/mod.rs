//! Playgen: autoregressive game-continuation model ("world sampler" for IS-DD).
//!
//! A causal transformer over tokenized auctions + plays, trained by teacher
//! forcing on self-play games. Sampling a continuation to the end of the deal
//! reveals a full hidden-hand assignment, i.e. a determinized world drawn from
//! the (approximate) posterior p(hands | observed public history).

/// Empreinte des sources qui décident du comportement du sidecar playgen
/// (`src/playgen/` + `src/engine/`), calculée à la compilation par `build.rs`.
///
/// Le sidecar la publie sur son `GET /health` et le conteneur web compare avec
/// la sienne : deux valeurs identiques disent que les deux binaires ont été
/// construits sur les mêmes sources. C'est le contrôle qui manquait le
/// 2026-08-03, quand un commit titré « feat(elo) » a livré la contrainte belote
/// dans le sampler et laissé la prod 21 h sur un sidecar périmé.
///
/// **Ne couvre pas** les poids du checkpoint, les drapeaux de compilation ni la
/// version de CUDA — voir `build.rs` pour le périmètre exact et ses raisons.
pub const SURFACE: &str = env!("COLVER_PLAYGEN_SURFACE");

pub mod tokens;

#[cfg(feature = "rand")]
pub mod infer;

#[cfg(feature = "rand")]
pub mod analysis;

#[cfg(feature = "dmc_train")]
pub mod model;

#[cfg(feature = "dmc_train")]
pub mod gpu;

#[cfg(test)]
mod surface_tests {
    /// Une empreinte vide ou malformée passerait inaperçue : `_freshness`
    /// traiterait les deux côtés comme « inconnus » et /health se tairait pour
    /// toujours — une alerte morte est pire que pas d'alerte, parce qu'on croit
    /// l'avoir. Ce test est le garde-fou de ce silence-là.
    #[test]
    fn surface_is_a_well_formed_fingerprint() {
        let s = super::SURFACE;
        assert_eq!(s.len(), 16, "empreinte attendue sur 16 chiffres hex : {s:?}");
        assert!(
            s.chars().all(|c| c.is_ascii_hexdigit()),
            "empreinte non hexadécimale : {s:?}"
        );
        assert_ne!(s, "0000000000000000", "empreinte nulle — build.rs n'a rien lu");
    }
}
