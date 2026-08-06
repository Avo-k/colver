//! Jouer des milliers de donnes en lockstep, une passe GPU par carte.
//!
//! # Le problème que ça résout
//!
//! DouDou50 en inférence CPU est limité par la **bande passante mémoire**, pas
//! par le calcul : une passe avant lit 10,2 Mo de poids pour faire 5,1 MFLOP,
//! soit **0,5 FLOP par octet**. Un seul fil sature déjà la DRAM, d'où le
//! constat mesuré : paralléliser sur rayon rend **1,4× pour 30× le CPU**.
//!
//! Grouper `B` déroulements change la nature du calcul : les poids sont lus
//! **une fois pour B passes**, donc l'intensité arithmétique est multipliée par
//! B et le produit matrice-vecteur devient un produit matrice-matrice. À
//! B = 5 000, ce n'est plus le même problème — c'est celui pour lequel un GPU
//! existe.
//!
//! # Comment le lockstep est possible
//!
//! Les déroulements sont indépendants, et **une fois l'enchère finie, ils ont
//! tous exactement 32 cartes à jouer**. Ils avancent donc en phase : à l'étape
//! `k`, chaque lane pose sa `k`-ième carte. Les décisions de l'étape `k` sont
//! indépendantes entre lanes, donc une seule passe avant les couvre toutes.
//!
//! L'enchère, elle, ne se met pas en lockstep aussi bien (les lanes en sortent
//! à des moments différents, et certaines finissent par une donne passée). Elle
//! reste sur CPU ici — c'est peu cher par appel, et la mesure dira si elle
//! devient le goulot une fois le jeu déporté.
//!
//! # Le contrôle obligatoire
//!
//! Deux pièges silencieux se cumulent sur ce chemin : l'orientation des
//! matrices (cf. [`DuelingQNet::from_raw_weights`]) et l'espace canonique des
//! couleurs (411 dims, cf. [`crate::agent::dmc::DmcPlayer`]). Se tromper sur
//! l'un ou l'autre ne lève **aucune erreur** : le réseau rend une carte légale
//! et sans rapport, ce qui se lit comme un joueur un peu faible. D'où
//! [`GpuRollout::check_against_cpu`], qui compare les Q du GPU à ceux de
//! `DmcNet::evaluate` sur les mêmes positions, et que tout appelant doit passer
//! avant de croire un chiffre.

use candle_core::{Device, Tensor};

use crate::agent::models::DmcWeights;
use crate::agent::MatchContext;
use crate::dmc_candle::DuelingQNet;
use crate::dmc_obs::{self, OBS_DIM_TR};
use crate::state::{GameState, Phase};

/// Une position à dérouler, avec le suivi public qui va avec.
pub struct Lane {
    pub state: GameState,
    pub ctx: MatchContext,
}

pub struct GpuRollout {
    net: DuelingQNet,
    device: Device,
    obs_dim: usize,
    canonical: bool,
}

impl GpuRollout {
    /// Charge les poids d'inférence sur GPU (CPU en repli si CUDA est absent).
    pub fn new(weights: &DmcWeights, residual: bool) -> candle_core::Result<Self> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let net = DuelingQNet::from_raw_weights(
            weights.floats(),
            weights.hidden,
            weights.obs_dim,
            residual,
            &device,
        )?;
        Ok(GpuRollout {
            net,
            device,
            obs_dim: weights.obs_dim,
            canonical: weights.obs_dim == OBS_DIM_TR,
        })
    }

    pub fn device_is_cuda(&self) -> bool {
        self.device.is_cuda()
    }

    /// Déroule toutes les lanes jusqu'au bout de la phase de jeu.
    ///
    /// Les lanes déjà terminées (donne passée) sont ignorées. À chaque étape,
    /// **une seule passe avant** couvre toutes les lanes vivantes.
    pub fn play_out(&self, lanes: &mut [Lane]) -> candle_core::Result<()> {
        let mut active: Vec<usize> =
            (0..lanes.len()).filter(|&i| lanes[i].state.phase == Phase::Playing).collect();
        if active.is_empty() {
            return Ok(());
        }

        let mut obs = vec![0.0f32; active.len() * self.obs_dim];
        // Ordre canonique par lane : il dépend de la main initiale et de l'atout,
        // donc il est constant sur toute la donne — calculé une fois par étape
        // quand même, parce qu'il dépend aussi du siège qui joue.
        let mut orders: Vec<[u8; 4]> = vec![[0, 1, 2, 3]; active.len()];
        let mut masks: Vec<u32> = vec![0; active.len()];

        // Au plus 32 cartes ; la garde évite qu'un bug de phase boucle sans fin.
        for _ in 0..32 {
            if active.is_empty() {
                break;
            }
            obs.truncate(active.len() * self.obs_dim);
            obs.resize(active.len() * self.obs_dim, 0.0);
            orders.resize(active.len(), [0, 1, 2, 3]);
            masks.resize(active.len(), 0);

            for (slot, &i) in active.iter().enumerate() {
                let lane = &lanes[i];
                let legal = lane.state.legal_actions() as u32;
                if self.canonical {
                    dmc_obs::write_observation_tr(
                        &mut obs,
                        slot * self.obs_dim,
                        &lane.state,
                        &lane.ctx.tracking,
                    );
                    let order = dmc_obs::current_player_order(&lane.state, &lane.ctx.tracking);
                    masks[slot] = dmc_obs::cardset_to_canonical(legal, &order);
                    orders[slot] = order;
                } else {
                    dmc_obs::write_observation(
                        &mut obs,
                        slot * self.obs_dim,
                        &lane.state,
                        &lane.ctx.tracking,
                    );
                    masks[slot] = legal;
                    orders[slot] = [0, 1, 2, 3];
                }
            }

            // La passe qui justifie tout le module.
            let input = Tensor::from_slice(&obs, (active.len(), self.obs_dim), &self.device)?;
            let q = self.net.forward(&input)?.to_vec2::<f32>()?;

            // L'argmax masqué revient sur CPU : 32 flottants par lane, donc le
            // transfert est négligeable, et le passage canonique → physique est
            // de toute façon propre à chaque lane.
            let mut still_active = Vec::with_capacity(active.len());
            for (slot, &i) in active.iter().enumerate() {
                let row = &q[slot];
                let mut best = 0u8;
                let mut best_q = f32::NEG_INFINITY;
                let mut m = masks[slot];
                while m != 0 {
                    let c = m.trailing_zeros() as u8;
                    if row[c as usize] > best_q {
                        best_q = row[c as usize];
                        best = c;
                    }
                    m &= m - 1;
                }
                let action = if self.canonical {
                    dmc_obs::card_to_physical(best, &orders[slot])
                } else {
                    best
                };
                let before = lanes[i].state;
                lanes[i].ctx.track(&before, action);
                lanes[i].state.step(action);
                if lanes[i].state.phase == Phase::Playing {
                    still_active.push(i);
                }
            }
            active = still_active;
        }
        Ok(())
    }

    /// **Le contrôle qui doit précéder toute mesure.** Rejoue `positions` carte
    /// par carte avec le réseau CPU de référence et compare les Q.
    ///
    /// Rend l'écart absolu maximal observé. Les deux chemins ne sont **pas
    /// bit-à-bit identiques** — l'ordre de réduction d'un produit matriciel
    /// diffère de celui d'un produit matrice-vecteur — donc on attend un écart
    /// de l'ordre de 1e-3, pas zéro. Ce qu'on cherche, c'est l'écart
    /// *structurel* : une matrice transposée ou un ordre canonique manqué
    /// donnent des valeurs sans rapport, pas un dernier chiffre qui bouge.
    pub fn check_against_cpu(
        &self,
        weights: &DmcWeights,
        residual: bool,
        positions: &[Lane],
    ) -> candle_core::Result<f32> {
        let mut cpu = weights.instantiate(residual);
        let mut worst = 0.0f32;
        let mut obs = vec![0.0f32; self.obs_dim];
        for lane in positions.iter().filter(|l| l.state.phase == Phase::Playing) {
            if self.canonical {
                dmc_obs::write_observation_tr(&mut obs, 0, &lane.state, &lane.ctx.tracking);
            } else {
                dmc_obs::write_observation(&mut obs, 0, &lane.state, &lane.ctx.tracking);
            }
            let want = cpu.evaluate(&obs);
            let input = Tensor::from_slice(&obs, (1, self.obs_dim), &self.device)?;
            let got = self.net.forward(&input)?.to_vec2::<f32>()?;
            for c in 0..32 {
                worst = worst.max((want[c] - got[0][c]).abs());
            }
        }
        Ok(worst)
    }
}
