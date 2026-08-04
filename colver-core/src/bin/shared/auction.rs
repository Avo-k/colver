//! Fabrique et manipulation d'**enchères**, partagé par les binaires de mesure.
//!
//! Extrait de `bench_taker_position` quand `bench_prefix_label` en a eu besoin.
//! Le partage n'est pas cosmétique : les deux binaires doivent produire **le même**
//! préfixe pour une même donne, sinon la mesure B ne porte pas sur les enchères que
//! la mesure A a caractérisées. Une copie aurait dérivé au premier correctif.
//!
//! Inclus par `#[path = "shared/auction.rs"] mod auction;` — le fichier n'est pas dans
//! `src/bin/*.rs` donc Cargo ne le prend pas pour un binaire.
#![allow(dead_code)]

use colver_core::bid_eval::evaluate_for_trump;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::bidding::BID_PASS;
use colver_core::card::Suit;
use colver_core::dmc_obs::EnvTracking;
use colver_core::state::{GameState, Phase};

/// Position d'un siège dans l'ordre de parole : 0 = premier à parler = `(dealer+1)%4`,
/// 3 = le donneur, qui parle en dernier.
#[inline]
pub fn speak_pos(seat: u8, dealer: u8) -> usize {
    ((seat + 4 - dealer + 3) % 4) as usize
}

/// Le siège que la construction ferait annoncer : dans le camp que `dd_pts` désigne,
/// celui des deux partenaires dont la main est la plus forte à cet atout.
///
/// Égalité départagée par la **position de parole**, pas par l'indice de siège : à
/// force égale c'est celui qui parle le premier qui prend l'initiative sur la couleur.
pub fn constructed_seat(hands: &[u32; 4], dealer: u8, trump: u8, side: u8) -> u8 {
    let suit = Suit::from_u8(trump);
    let (a, b) = if side == 0 { (0u8, 2u8) } else { (1u8, 3u8) };
    let (ea, eb) = (
        evaluate_for_trump(hands[a as usize], suit),
        evaluate_for_trump(hands[b as usize], suit),
    );
    match ea.cmp(&eb) {
        std::cmp::Ordering::Greater => a,
        std::cmp::Ordering::Less => b,
        std::cmp::Ordering::Equal => {
            if speak_pos(a, dealer) < speak_pos(b, dealer) { a } else { b }
        }
    }
}

/// Camp qui peut tenir un contrat à cet atout. Les points cartes sont à somme
/// constante, donc un seul des deux le peut — c'est ce qui donne le bénéfice d'un
/// `[u8;8]` au prix d'un `[u8;4]`.
#[inline]
pub fn dd_side(ns_pts: u8) -> u8 {
    if ns_pts > 81 { 0 } else { 1 }
}

/// Rejoue `prefix` tel quel, puis laisse v6 mener l'enchère sous `mask` jusqu'au bout.
///
/// `mask = u64::MAX` et `prefix` vide donnent l'enchère libre — le témoin. Un `prefix`
/// non vide sert à l'épluchage : on remet v6 dans une situation qu'il a réellement
/// traversée, à une action près.
pub fn run_v6(
    hands: &[u32; 4],
    dealer: u8,
    mask: u64,
    prefix: &[u8],
    net: &mut BidNet,
    obs: &mut Vec<f32>,
) -> Vec<u8> {
    let mut g = GameState::new(dealer, *hands);
    let mut tr = EnvTracking::new();
    tr.dealer = dealer;
    let dim = net.obs_dim();
    let mut out: Vec<u8> = Vec::with_capacity(12);

    for &a in prefix {
        if g.phase != Phase::Bidding {
            return out;
        }
        tr.track_action(&g, a);
        g.step(a);
        out.push(a);
    }

    while g.phase == Phase::Bidding {
        let legal = g.legal_actions() & mask;
        // `PASS` est dans les deux, donc l'intersection n'est jamais vide.
        debug_assert!(legal & (1 << BID_PASS) != 0);
        // Donnes isolées : 0-0. Même contexte de score que le corpus de référence et
        // que la génération de la couche.
        obs.clear();
        obs.resize(dim, 0.0);
        bid_obs::write_bid_observation_dim(obs, 0, &g, &tr.bid_history, 0, 0, dim);
        let action = net.best_action_fast(obs, legal);
        tr.track_action(&g, action);
        g.step(action);
        out.push(action);
        if out.len() > 40 {
            break; // garde-fou
        }
    }
    out
}

/// Rejoue `prefix`, puis ferme l'enchère avec des passes.
///
/// C'est la variante « troncature sèche » : on **affirme** que personne ne relance,
/// au lieu de le demander à v6. Moins cher, et surtout **déterministe sur l'atout** —
/// le contrat est forcément l'annonce précédente. Ce que ça coûte en réalisme se lit
/// en comparant avec `run_v6` sur le même préfixe.
pub fn close_with_passes(hands: &[u32; 4], dealer: u8, prefix: &[u8]) -> Vec<u8> {
    let mut g = GameState::new(dealer, *hands);
    let mut out: Vec<u8> = Vec::with_capacity(12);
    for &a in prefix {
        if g.phase != Phase::Bidding {
            return out;
        }
        g.step(a);
        out.push(a);
    }
    while g.phase == Phase::Bidding && out.len() <= 40 {
        g.step(BID_PASS);
        out.push(BID_PASS);
    }
    out
}

#[inline]
pub fn is_bid(a: u8) -> bool {
    (1..=40).contains(&a)
}

/// L'annonce à l'indice `idx` est-elle une **relance** de son auteur — a-t-il déjà
/// annoncé plus tôt ?
///
/// Le siège se déduit sans rejeu : une enchère avance d'un siège par action, donc
/// `seat(i) = (dealer + 1 + i) % 4`. Les tours de parole d'un même siège sont donc
/// exactement les indices congrus à `idx` modulo 4.
pub fn is_raise(actions: &[u8], idx: usize) -> bool {
    let mut j = idx;
    while j >= 4 {
        j -= 4;
        if is_bid(actions[j]) {
            return true;
        }
    }
    false
}

