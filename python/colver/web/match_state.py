"""La partie : une suite de donnes jouée jusqu'à un score cible.

Une *donne* est ce que le reste du serveur appelle un « game » (une ligne de la
table `games`, un `PlaySession`, un identifiant partageable). Une *partie* les
enchaîne : le donneur tourne, les scores s'additionnent, et le premier camp qui
atteint la cible l'emporte. `target = 0` = donne unique, c'est-à-dire le
comportement historique du site — le reste du code n'a alors rien de spécial à
faire, il y a toujours un `Match`, il se termine juste après une donne.

Le score courant de la partie n'est pas qu'un affichage : il est passé aux bots
(`AgentTable.set_scores`) et le bidder v6, entraîné avec l'observation
score-aware, annonce différemment à 900-200 qu'à 0-0.
"""

import random

# Cibles proposées côté client. 0 = une donne, sans suite.
TARGETS = (0, 1000, 2000)
DEFAULT_TARGET = 0


def normalize_target(value) -> int:
    """Cible demandée par le client → cible valide (0 par défaut)."""
    try:
        target = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TARGET
    return target if target in TARGETS else DEFAULT_TARGET


class Match:
    """Le déroulé d'une partie : scores cumulés, donneur, donnes jouées.

    Les scores sont indexés par équipe **physique** (`[NS, EW]`), comme partout
    ailleurs côté serveur ; la rotation vers le repère du spectateur est faite
    au moment de la diffusion (cf. `rooms.rotate_state`).
    """

    def __init__(self, target=DEFAULT_TARGET, dealer=None, match_id=None):
        self.target = normalize_target(target)
        self.id = match_id
        self.dealer = random.randrange(4) if dealer is None else int(dealer) % 4
        self.totals = [0, 0]
        self.deals = []      # [{"game_id", "scores": [ns, ew], "dealer"}]
        self.deal_no = 1     # numéro (1-based) de la donne en cours

    @classmethod
    def restore(cls, target, totals, deals, dealer, match_id=None):
        """Reconstruire une partie interrompue, prête à donner la donne suivante.

        `totals` est le cumul stocké (`matches.points_ns/ew`) : le score marqué
        d'une donne n'est enregistré nulle part donne par donne, seulement
        additionné là — `games.points_ns/ew` sont les points *cartes*, une autre
        échelle. Les donnes déjà jouées reviennent donc sans leur score
        (`"scores": None`, plutôt qu'un chiffre de la mauvaise unité) ; elles ne
        servent plus qu'à compter et à numéroter.

        `dealer` est le donneur de la donne **à venir**, calculé par l'appelant :
        lui seul sait si la coupure a laissé une donne en plan (le même joueur
        redonne) ou pas (le donneur passe à gauche).
        """
        match = cls(target=target, dealer=dealer, match_id=match_id)
        match.totals = [int(totals[0]), int(totals[1])]
        match.deals = [
            {"game_id": d["game_id"], "scores": None, "dealer": d["dealer"]}
            for d in deals
        ]
        match.deal_no = len(match.deals) + 1
        return match

    # ----- déroulé -----

    @property
    def is_match(self) -> bool:
        """Vrai quand plusieurs donnes s'enchaînent (cible > 0)."""
        return self.target > 0

    def record(self, game_id, rewards) -> bool:
        """Ajouter le résultat d'une donne terminée aux scores de la partie.

        Rend False si cette donne était déjà comptée : plusieurs chemins mènent
        à la fin d'une donne (le coup humain terminal, puis le tour des bots),
        et une double addition fausserait toute la partie.
        """
        if any(d["game_id"] == game_id for d in self.deals):
            return False
        scores = [int(rewards[0]), int(rewards[1])]
        self.totals[0] += scores[0]
        self.totals[1] += scores[1]
        self.deals.append({
            "game_id": game_id,
            "scores": scores,
            "dealer": self.dealer,
        })
        return True

    def next_deal(self):
        """Préparer la donne suivante : le donneur passe au joueur de gauche."""
        self.dealer = (self.dealer + 1) % 4
        self.deal_no = len(self.deals) + 1

    @property
    def finished(self) -> bool:
        """La partie est-elle jouée ?

        Une donne unique s'arrête d'elle-même. En partie, il faut qu'un camp ait
        atteint la cible **et** que les deux ne soient pas à égalité : les deux
        camps marquent à chaque donne, ils peuvent franchir les 1000 ensemble, et
        à égalité parfaite la partie continue.
        """
        if not self.is_match:
            return len(self.deals) >= 1
        a, b = self.totals
        return max(a, b) >= self.target and a != b

    @property
    def winner(self):
        """Camp vainqueur (0 = NS, 1 = EW), ou None tant que rien n'est joué."""
        if not self.finished:
            return None
        a, b = self.totals
        if a == b:
            return None
        return 0 if a > b else 1

    # ----- diffusion -----

    def payload(self):
        """Blob envoyé au client (repère physique — à faire tourner en salon)."""
        return {
            "id": self.id,
            "target": self.target,
            "totals": list(self.totals),
            "deal_no": self.deal_no,
            "deals": [dict(d) for d in self.deals],
            "finished": self.finished,
            "winner": self.winner,
        }


# ===== Ce qu'un temps de jeu écoulé coûte =====
#
# Ici plutôt que dans `rooms`, parce que **deux modules en dépendent** et qu'ils
# ne peuvent pas s'importer l'un l'autre : le pilote de salon applique les
# seuils en direct (`SeatClock`), et le classement les relit après coup depuis
# le journal des donnes. Les dupliquer, c'est se garantir qu'ils divergeront le
# jour où l'un des deux bougera.
#
# Les deux seuils ne mesurent pas la même chose, et l'ordre compte :
#
# - **consécutifs** = une absence. Six d'affilée rendent le siège à l'IA, et
#   c'est une **défaite** pour la personne — sinon partir serait gratuit.
# - **cumulés** = une délégation : laisser filer la pendule pour voir jouer le
#   bot à sa place. La partie cesse alors de compter **pour ce joueur-là**,
#   personne d'autre n'étant lésé.
#
# Un siège qui a forfait dépasse forcément les deux : c'est la défaite qui
# l'emporte, la sanction la plus faible ne doit pas absorber la plus forte.
MISSES_TO_FORFEIT = 6
MISSES_TOTAL_UNRATED = 3
